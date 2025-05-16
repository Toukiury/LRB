from data_trace.data_trace import DataTrace
from model.models import ParrotModel, LightGBMModel
from model import device_manager
from utils.aligner import ShiftAligner, NormalAligner
from cache.cache import Cache, BoostCache, DumpCache
from cache.evict import *
from cache.evict.algorithms import SimpleGuardLRBAlgorithm
from cache.hash import ShiftHashFunction, BrightKiteHashFunction, CitiHashFunction
from functools import partial
from typing import Tuple
from prettytable import PrettyTable
import tqdm
import copy
import argparse
import os
import pickle
import json
from pathos.multiprocessing import ProcessingPool as Pool

def process_cache(cache):
    with DataTrace(file_path) as trace:
        while not trace.done():
            pc, address = trace.next()
            cache.access(pc, address)
        return cache.stat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xalanc')
    parser.add_argument("--test_all", action='store_true')

    parser.add_argument("--device", type=str, default='cpu')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--oracle', action='store_true')
    mode_group.add_argument('--real', action='store_true')
    mode_group.add_argument('--lrb', action='store_true', help='使用LRB算法')
    mode_group.add_argument('--lrb_fr', action='store_true', help='使用LRB算法和F&R框架')
    mode_group.add_argument('--lrbcomplete', action='store_true', help='使用LRBComplete完整实现')
    mode_group.add_argument('--lrbcomplete_fr', action='store_true', help='使用LRBComplete与F&R框架')
    mode_group.add_argument('--guard_lrb', action='store_true', help='使用SimpleGuardLRB算法')

    parser.add_argument('--pred', nargs='+', default='none', choices=['parrot', 'pleco', 'popu', 'pleco-bin', 'gbm', 'lrb', 'oracle_bin', 'oracle_dis'])

    parser.add_argument("--noise_type", type=str, default='logdis', choices=['dis', 'bin', 'logdis'])

    parser.add_argument("--dump_file", action='store_true')
    parser.add_argument("--output_root_dir", type=str, default='res')

    parser.add_argument("--verbose", action='store_true')

    parser.add_argument("--boost", action='store_true')
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--boost_fr", action='store_true')
    parser.add_argument("--boost_preds_dir", type=str, default='boost_traces')

    parser.add_argument("--model_fraction", type=str, default='1')
    parser.add_argument("--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("--parrot_config_path", type=str, default='checkpoints/parrot/model_config.json')
    parser.add_argument("--lightgbm_config_path", type=str, default='checkpoints/lightgbm/model_config.json')

    # LRB相关参数
    parser.add_argument("--memory_window", type=int, default=1000000, help="LRB算法内存窗口大小")
    parser.add_argument("--relaxation_factor", type=float, default=10.0, help="LRB放松因子（决定对象价值的影响力）")
    parser.add_argument("--admission_size", type=int, default=None, help="LRB准入队列大小，默认为associativity/4")
    parser.add_argument("--lrb_only", action="store_true", help="只使用基本LRB算法，不包含其他基于LRB的变体")
    parser.add_argument("--disable_admission", action="store_true", help="禁用LRB准入策略")
    parser.add_argument("--disable_edc", action="store_true", help="禁用LRB的EDC特征")
    parser.add_argument("--debug", action="store_true", help="启用调试模式，打印详细信息")
    
    # SimpleGuardLRB相关参数
    parser.add_argument("--relax_times", type=int, default=0, help="SimpleGuardLRB的relax_times参数值，控制保护机制触发条件")
    parser.add_argument("--relax_prob", type=float, default=0.0, help="SimpleGuardLRB的relax_prob参数值，控制保护机制触发概率")
    parser.add_argument("--follow_if_guarded", action="store_true", help="SimpleGuardLRB在保护模式下是否仍然遵循预测结果")
    
    # 添加SimpleGuardLRB特有参数
    parser.add_argument("--simple_relax_times", type=int, default=0, help="SimpleGuardLRB的relax_times参数值，控制保护机制触发条件")
    parser.add_argument("--simple_relax_prob", type=float, default=0.0, help="SimpleGuardLRB的relax_prob参数值，控制保护机制触发概率")
    
    # 添加静默参数，减少冗余输出
    parser.add_argument("--quiet", action='store_true', help="减少输出冗余信息，只显示最终结果")

    args = parser.parse_args()
    
    # 设置进度条禁用标志
    disable_progress = args.quiet if hasattr(args, 'quiet') else False
    
    file_path = f'traces/{args.dataset}/{args.dataset}_test.csv'
    if args.test_all:
        if args.dataset == 'brightkite' or args.dataset == 'citi':
            file_path = f'traces/{args.dataset}/{args.dataset}_all.csv'
    if args.dataset == 'brightkite':
        cache_line_size = 1
        capacity = 1000
        associativity = 10
        align_type = NormalAligner
        hash_type = BrightKiteHashFunction
    elif args.dataset == 'citi':
        cache_line_size = 1
        capacity = 1200
        associativity = 100
        align_type = NormalAligner
        hash_type = CitiHashFunction
    else:
        cache_line_size = 64
        capacity = 2097152
        associativity = 16
        align_type = ShiftAligner
        hash_type = ShiftHashFunction

    # 根据命令行参数设置是否包含LRB变体
    if hasattr(args, 'lrb_only') and args.lrb_only:
        PredictAlgorithmFactory.include_lrb_variants = False
    else:
        PredictAlgorithmFactory.include_lrb_variants = True

    this_preds = []
    real_predictors_type = ['parrot', 'pleco', 'popu', 'pleco-bin', 'gbm', 'lrb']
    oracle_predictors_type = ['oracle_bin', 'oracle_dis']
    input_preds = args.pred
    if 'none' in input_preds:
        if args.real:
            this_preds = real_predictors_type
        else:
            this_preds = oracle_predictors_type
    else:
        this_preds = input_preds
    
    ###########################################################
    parrot_gen = gbm_gen = lrb_gen = None
    ckpt_root_dir = args.checkpoints_root_dir
    if 'parrot' in this_preds:
        this_dir = os.path.join(ckpt_root_dir, 'parrot', args.dataset, args.model_fraction)
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'best.ckpt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')
        print('Parrot Model Checkpoint:', this_ckpt_path)
        print('Parrot Model Fraction:', args.model_fraction)
        print('Parrot Model Config Path:', args.parrot_config_path)
        parrot_gen = lambda : ParrotModel.from_config(args.parrot_config_path, this_ckpt_path)
    if 'gbm' in this_preds:
        with open(args.lightgbm_config_path, "r") as f:
            model_config = json.load(f)
            deltanums = model_config['delta_nums']
            edcnums = model_config['edc_nums']

        this_dir = os.path.join(ckpt_root_dir, 'lightgbm', args.dataset, args.model_fraction)
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'{args.dataset}_{args.model_fraction}_{deltanums}_{edcnums}.txt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')

        threshold = 0.5
        threshold_path = os.path.join(this_dir, 'threshold')
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as file:
                content = file.read().strip()
                threshold = float(content)
        print(f'LightGBM: Fraction [{args.model_fraction}], Threshold [{threshold}], Model Checkpoint[{this_ckpt_path}], Delta[{deltanums}], EDC[{edcnums}]')
        gbm_gen = lambda : LightGBMModel.from_config(deltanums, edcnums, this_ckpt_path, threshold)
    
    if 'lrb' in this_preds:
        # 可以复用GBM的模型加载方式和特征，不同之处在于LRB的内存窗口管理
        with open(args.lightgbm_config_path, "r") as f:
            model_config = json.load(f)
            deltanums = model_config['delta_nums']
            edcnums = model_config['edc_nums']

        this_dir = os.path.join(ckpt_root_dir, 'lightgbm', args.dataset, args.model_fraction)
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'{args.dataset}_{args.model_fraction}_{deltanums}_{edcnums}.txt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')

        threshold = 0.5
        threshold_path = os.path.join(this_dir, 'threshold')
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as file:
                content = file.read().strip()
                threshold = float(content)
        print(f'LRB: Fraction [{args.model_fraction}], Memory Window [{args.memory_window}], Threshold [{threshold}], Model Checkpoint[{this_ckpt_path}], Delta[{deltanums}], EDC[{edcnums}]')
        lrb_gen = lambda : LightGBMModel.from_config(deltanums, edcnums, this_ckpt_path, threshold)
    
    if args.lrbcomplete or args.lrbcomplete_fr:
        # 加载LightGBM模型用于LRB算法
        with open(args.lightgbm_config_path, "r") as f:
            model_config = json.load(f)
            deltanums = model_config['delta_nums']
            edcnums = model_config['edc_nums']

        this_dir = os.path.join(ckpt_root_dir, 'lightgbm', args.dataset, args.model_fraction)
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'{args.dataset}_{args.model_fraction}_{deltanums}_{edcnums}.txt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')

        threshold = 0.5
        threshold_path = os.path.join(this_dir, 'threshold')
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as file:
                content = file.read().strip()
                threshold = float(content)
        
        # 设置LRB参数
        memory_window = args.memory_window
        relaxation_factor = args.relaxation_factor
        admission_size = args.admission_size if args.admission_size else associativity // 4
        enable_admission = not args.disable_admission
        enable_edc = not args.disable_edc
        debug_mode = args.debug if hasattr(args, 'debug') else False
        
        # 加载GBM模型（用于LRB的预测组件）
        lrb_model = LightGBMModel.from_config(deltanums, edcnums, this_ckpt_path, threshold)
        
        print(f'LRBComplete: Dataset [{args.dataset}], Fraction [{args.model_fraction}]')
        print(f'LRBComplete: Memory Window [{memory_window}], Relaxation Factor [{relaxation_factor}], Admission Size [{admission_size}]')
        print(f'LRBComplete: Threshold [{threshold}], Model Checkpoint [{this_ckpt_path}]')
        print(f'LRBComplete: Delta [{deltanums}], EDC [{edcnums}]')
        print(f'LRBComplete: Enable Admission [{enable_admission}], Enable EDC [{enable_edc}], Debug Mode [{debug_mode}]')
        
        # 创建LRBComplete算法实例
        lrb_algorithm = partial(LRBAlgorithm, memory_window=memory_window, relaxation_factor=relaxation_factor, 
                              admission_size=admission_size, enable_admission=enable_admission, enable_edc=enable_edc,
                              debug_mode=debug_mode)
        
        # 创建测试算法列表
        algorithms_to_test = []
        
        # 添加LRBComplete算法
        algorithms_to_test.append(PredictAlgorithmFactory.generate_predictive_algorithm(lrb_algorithm, 'LRB', shared_model=lrb_model))
        
        # 在F&R框架下评估LRB
        if args.lrbcomplete_fr:
            algorithms_to_test.extend([
                RandAlgorithm(associativity),
                LRUAlgorithm(associativity),
                MarkerAlgorithm(associativity),
                # 添加标准LRB预测器（作为组件）与不同替换策略的组合
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'LRB', shared_model=lrb_model, memory_window=memory_window),
                PredictAlgorithmFactory.generate_predictive_algorithm(Mark0, 'LRB', shared_model=lrb_model, memory_window=memory_window),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'LRB', shared_model=lrb_model, memory_window=memory_window),
            ])
            
            # 添加OPT作为基准
            opt_algorithm = PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleDis')
            algorithms_to_test.append(opt_algorithm)
        
        # 创建缓存实例
        caches = []
        for algorithm in algorithms_to_test:
            cache = Cache(file_path, align_type, algorithm, hash_type, cache_line_size, capacity, associativity)
            caches.append(cache)
        
        # 运行测试
        with DataTrace(file_path) as trace:
            with tqdm.tqdm(desc="Running LRBComplete test", disable=disable_progress) as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    for cache in caches:
                        cache.access(pc, address)
                    pbar.update(1)
        
        # 创建结果表格
        table = PrettyTable()
        table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate"]
        
        # 遍历所有缓存算法，获取统计信息
        for i, cache in enumerate(caches):
            hit, miss, total, rate = cache.stat()
            algorithm_name = "OPT" if i == len(caches) - 1 and args.lrbcomplete_fr else "LRB"
            if i > 0 and i < len(caches) - 1 and args.lrbcomplete_fr:
                algorithm_name = ["Rand", "LRU", "Marker", "LRB-Base", "Mark0-LRB", "Guard-LRB"][i-1]
            table.add_row([algorithm_name, hit, miss, total, f"{rate:.4f}"])
        
        # 输出结果表格
        print(table)
        
        # 保存结果
        res_dir = os.path.join(args.output_root_dir, args.dataset, args.model_fraction)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        
        file_suffix = "fr" if args.lrbcomplete_fr else "solo"
        result_file = os.path.join(res_dir, f"lrbcomplete_{file_suffix}.csv")
        with open(result_file, "w", encoding="utf-8") as file:
            file.write(table.get_csv_string())
        
        print(f"结果已保存到: {result_file}")
        
        # 直接退出，不执行后面的代码
        import sys
        sys.exit(0)
    
    if args.guard_lrb:
        # 加载LightGBM模型用于SimpleGuardLRB算法
        with open(args.lightgbm_config_path, "r") as f:
            model_config = json.load(f)
            deltanums = model_config['delta_nums']
            edcnums = model_config['edc_nums']

        this_dir = os.path.join(ckpt_root_dir, 'lightgbm', args.dataset, args.model_fraction)
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'{args.dataset}_{args.model_fraction}_{deltanums}_{edcnums}.txt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')
        
        # 获取阈值
        threshold = 0.5
        threshold_path = os.path.join(this_dir, 'threshold')
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as file:
                content = file.read().strip()
                threshold = float(content)
        
        # 设置SimpleGuardLRB参数
        memory_window = args.memory_window
        admission_size = args.admission_size if args.admission_size else associativity // 4
        enable_admission = not args.disable_admission if hasattr(args, 'disable_admission') else True
        debug_mode = args.debug if hasattr(args, 'debug') else False
        
        # SimpleGuardLRB特有参数
        relax_times = args.simple_relax_times if hasattr(args, 'simple_relax_times') else 0
        relax_prob = args.simple_relax_prob if hasattr(args, 'simple_relax_prob') else 0.0
        follow_if_guarded = args.follow_if_guarded if hasattr(args, 'follow_if_guarded') else False
        
        print(f'SimpleGuardLRB: Dataset [{args.dataset}], Fraction [{args.model_fraction}]')
        print(f'SimpleGuardLRB: Memory Window [{memory_window}]')
        print(f'SimpleGuardLRB: Threshold [{threshold}], Model Checkpoint [{this_ckpt_path}]')
        print(f'SimpleGuardLRB: Delta [{deltanums}], EDC [{edcnums}]')
        print(f'SimpleGuardLRB: Enable Admission [{enable_admission}], Debug Mode [{debug_mode}]')
        print(f'SimpleGuardLRB: Relax Times [{relax_times}], Relax Probability [{relax_prob}], Follow If Guarded [{follow_if_guarded}]')
        
        # 创建LightGBM模型
        lrb_model = LightGBMModel.from_config(deltanums, edcnums, this_ckpt_path, threshold)
        
        # 创建SimpleGuardLRB算法工厂函数
        evict_type = partial(SimpleGuardLRBAlgorithm,
            evictor_type=BinaryEvictor,
            predictor_type=partial(LRBPredictor, shared_model=lrb_model),
            memory_window=memory_window,
            admission_size=admission_size,
            enable_admission=enable_admission,
            debug_mode=debug_mode,
            relax_times=relax_times,
            relax_prob=relax_prob,
            follow_if_guarded=follow_if_guarded
        )
        
        # 创建缓存 - 使用标准Cache类
        cache = Cache(file_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity)
        
        # 运行评测
        with DataTrace(file_path) as trace:
            with tqdm.tqdm(desc="SimpleGuardLRB缓存评测", disable=disable_progress) as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    cache.access(pc, address)
                    pbar.update(1)
        
        # 打印结果
        hit, miss, total, rate = cache.stat()
        table = PrettyTable() 
        table.field_names = ["算法", "命中", "未命中", "总访问", "命中率"]
        
        # 定义算法名称
        algorithm_name = "SimpleGuardLRB"
        if relax_times > 0:
            algorithm_name += f"-RT{relax_times}"
        elif relax_prob > 0:
            algorithm_name += f"-RP{relax_prob:.2f}"
            
        table.add_row([algorithm_name, hit, miss, total, f"{rate:.4f}"])
        print(table)
        
        # 保存结果
        if args.dump_file:
            res_dir = os.path.join(args.output_root_dir, args.dataset, args.model_fraction)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            
            result_file = os.path.join(res_dir, f"simpleguardlrb_rt{relax_times}_rp{relax_prob}.csv")
            with open(result_file, "w", encoding="utf-8") as file:
                file.write(table.get_csv_string())
            
            print(f"结果已保存到: {result_file}")
        
        # 直接退出，不执行后面的代码
        import sys
        sys.exit(0)
    
    print("Benchmark: Use Predictor:", this_preds)
    print('Benchmark: Use Trace:', file_path)
    if args.dump_file:
        print('Benchmark: Output Path:', args.output_root_dir)
    else:
        print('Benchmark: Only print')
    if args.boost:
        if args.real:
            print('Benchmark: Use Boost Trace Prediction')
        else:
            print('Benchmark: Use MultiProcess Boost')
    if args.boost_fr:
        print('Benchmark: Enable F&R Boost')
    device = args.device
    device_manager.set_device(device)

    online_types = [
        RandAlgorithm,
        LRUAlgorithm,
        MarkerAlgorithm,
    ]

    sorted = False
    verbose = args.verbose

    func_dict = {}
    def register_func(this_partial, noise, baseline=False):
        if baseline:
            func_dict['OPT'] = {}
            func_dict['OPT'][0] = this_partial
        else:
            pretty_name = pretty_print(this_partial, verbose)
            if pretty_name not in func_dict:
                func_dict[pretty_name] = {}
            func_dict[pretty_name][noise] = this_partial
    
    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleDis'), 0, True)

    combiner_types = []

    boost_preds_dict = {}

    if not os.path.exists(args.boost_preds_dir):
        os.makedirs(args.boost_preds_dir)
    def boost_generate_prediction(pred_type, **kwargs):
        pred_algorithm = PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, pred_type, **kwargs)

        if args.test_all and (args.dataset == 'brightkite' or args.dataset == 'citi'):
            pred_pickle_path = os.path.join(args.boost_preds_dir, f'{args.dataset}_all_{pred_type}_{args.model_fraction}.pkl')
        else:
            pred_pickle_path = os.path.join(args.boost_preds_dir, f'{args.dataset}_{pred_type}_{args.model_fraction}.pkl')
        if not os.path.exists(pred_pickle_path):
            if not args.quiet:
                print(f'Boost Prediction: Generating Prediction for {pred_type}, Path: {pred_pickle_path}')
            if pred_type.endswith('State'):
                is_state = True
            else:
                is_state = False
            dump_cache = DumpCache(is_state, file_path, align_type, pred_algorithm, hash_type, cache_line_size, capacity, associativity)
            with DataTrace(file_path) as trace:
                with tqdm.tqdm(desc="Producing cache on Boost Prediction", disable=disable_progress) as pbar:
                    while not trace.done():
                        pc, address = trace.next()
                        dump_cache.simulate(pc, address)
                        pbar.update(1) 
            lst = dump_cache.dump()
            with open(pred_pickle_path, 'wb') as f:
                pickle.dump(lst, f)
            return lst
        else:
            if not args.quiet:
                print(f'Boost Prediction: Find boost prediction for {pred_type}, Path: {pred_pickle_path}')
            with open(pred_pickle_path, 'rb') as f:
                lst = pickle.load(f)
                return lst

    if args.real:
        verbose = True

        ##########################################
        if 'parrot' in this_preds:
            if args.boost:
                boost_preds_dict['Parrot'] = boost_generate_prediction('Parrot', shared_model=parrot_gen())
                boost_preds_dict['Parrot-State'] = boost_generate_prediction('Parrot-State', associativity=associativity, shared_model=parrot_gen())

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(FollowerRobust, boost=args.boost_fr), 'Parrot-State', associativity=associativity, shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'Parrot', shared_model=parrot_gen()),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_gen()), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_gen()), MarkerAlgorithm]),
            ])


        ##########################################
        if 'pleco' in this_preds:
            if args.boost:
                boost_preds_dict['PLECO'] = boost_generate_prediction('PLECO')
                boost_preds_dict['PLECO-State'] = boost_generate_prediction('PLECO-State', associativity=associativity)

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(FollowerRobust, boost=args.boost_fr), 'PLECO-State', associativity=associativity),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'PLECO'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), MarkerAlgorithm]),
            ])

        ##########################################
        if 'popu' in this_preds:
            if args.boost:
                boost_preds_dict['POPU'] = boost_generate_prediction('POPU')
                boost_preds_dict['POPU-State'] = boost_generate_prediction('POPU-State', associativity=associativity)

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(FollowerRobust, boost=args.boost_fr), 'POPU-State', associativity=associativity),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'POPU'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), MarkerAlgorithm]),
            ])
        
        ##########################################
        if 'pleco-bin' in this_preds:
            if args.boost:
                boost_preds_dict['PLECO-Bin'] = boost_generate_prediction('PLECO-Bin', threshold=0.5)

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO-Bin', threshold=0.5),
                PredictAlgorithmFactory.generate_predictive_algorithm(Mark0, 'PLECO-Bin', threshold=0.5),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'PLECO-Bin', threshold=0.5),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'PLECO-Bin', threshold=0.5),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO-Bin', threshold=0.5), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO-Bin', threshold=0.5), MarkerAlgorithm]),
            ])

        ##########################################
        if 'gbm' in this_preds:
            if args.boost:
                boost_preds_dict['GBM'] = boost_generate_prediction('GBM', shared_model=gbm_gen())

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(Mark0, 'GBM', shared_model=gbm_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'GBM', shared_model=gbm_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'GBM', shared_model=gbm_gen()),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()), MarkerAlgorithm]),
            ])

        ##########################################
        if 'lrb' in this_preds:
            if args.boost:
                boost_preds_dict['LRB'] = boost_generate_prediction('LRB', shared_model=lrb_gen(), memory_window=args.memory_window)

            # 基本LRB算法
            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'LRB', shared_model=lrb_gen(), memory_window=args.memory_window),
            ])
            
            # 根据include_lrb_variants标志决定是否添加额外的LRB变体
            if PredictAlgorithmFactory.include_lrb_variants:
                online_types.extend([
                    PredictAlgorithmFactory.generate_predictive_algorithm(Mark0, 'LRB', shared_model=lrb_gen(), memory_window=args.memory_window),
                    PredictAlgorithmFactory.generate_predictive_algorithm(partial(SimpleGuardLRBAlgorithm, follow_if_guarded=False, relax_times=0, relax_prob=0), 'LRB', shared_model=lrb_gen(), memory_window=args.memory_window),
                    PredictAlgorithmFactory.generate_predictive_algorithm(partial(SimpleGuardLRBAlgorithm, follow_if_guarded=False, relax_times=5, relax_prob=0), 'LRB', shared_model=lrb_gen(), memory_window=args.memory_window),
                ])
                
                combiner_types.extend([
                    (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'LRB', shared_model=lrb_gen(), memory_window=args.memory_window), MarkerAlgorithm]),
                    (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'LRB', shared_model=lrb_gen(), memory_window=args.memory_window), MarkerAlgorithm]),
                    # 添加LRB&LRU组合算法
                    (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'LRB', shared_model=lrb_gen(), memory_window=args.memory_window), LRUAlgorithm]),
                ])

        ########################################################
        for online_type in online_types:
            register_func(online_type, 0)

        oracle_types = [
            # (PredictAlgorithm, "OracleDis")
        ]

        for oracle_type, pred_type_str in oracle_types:
            register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str), 0)

        for combiner, algs in combiner_types:
            if len(algs) > 1:
                this_partial = copy.deepcopy(combiner)
                this_partial.keywords['candidate_algorithms'] = algs
                register_func(this_partial, 0)
    else:
        noise_type = args.noise_type
        oracle_types = []
        combiner_types = []

        for online_type in online_types:
            register_func(online_type, 0)

        oracle_logdis_noise_mask = [0, 5, 10, 20, 30, 40, 50]
        oracle_dis_noise_mask = [0, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        oracle_bin_noise_mask = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        if 'oracle_dis' in this_preds:
            oracle_types.extend([
                (PredictAlgorithm, 'OracleDis'),
                (PredictiveMarker, 'OracleDis'),
                (LMarker, 'OracleDis'),
                (LNonMarker, 'OracleDis'),
                (partial(FollowerRobust, boost=args.boost_fr), 'OracleState'),
                (partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'OracleDis'),
                (partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'OracleDis'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), MarkerAlgorithm]),
            ])

        #####################################################
        if 'oracle_bin' in this_preds:
            oracle_types.extend([
                (PredictAlgorithm, 'OracleBin'),
                (Mark0, 'OracleBin'),
                (MarkAndPredict, 'OraclePhase'),
                (partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'OracleBin'),
                (partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'OracleBin'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), MarkerAlgorithm]),
            ])

        #####################################################
        for oracle_type, pred_type_str in oracle_types:
            if noise_type == 'dis':
                for noise in oracle_dis_noise_mask:
                    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, reuse_dis_noise_sigma=noise, lognormal=False, associativity=associativity), noise)
            elif noise_type == 'logdis':
                for noise in oracle_logdis_noise_mask:
                    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, reuse_dis_noise_sigma=noise, lognormal=True, associativity=associativity), noise)
            elif noise_type == 'bin':
                if pred_type_str == 'OracleBin' or pred_type_str == 'OraclePhase':
                    for noise in oracle_bin_noise_mask:
                        register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, bin_noise_prob=noise, associativity=associativity), noise)
            else:
                raise ValueError('Invalid noise type')

        def mask_combiner(noise):
            for combiner, algs in combiner_types:
                candidate_algorithms = []
                for alg in algs:
                    if isinstance(alg, Tuple):
                        alg_type, pred_type_str = alg
                        if noise_type == 'dis':
                            candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, reuse_dis_noise_sigma=noise, lognormal=False))
                        elif noise_type == 'logdis':
                            candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, reuse_dis_noise_sigma=noise, lognormal=True))
                        elif noise_type == 'bin':
                            if pred_type_str == 'OracleBin' or pred_type_str == 'OraclePhase':
                                candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, bin_noise_prob=noise))
                        else:
                            raise ValueError('Invalid noise type')
                    else:
                        candidate_algorithms.append(alg)
                
                if len(candidate_algorithms) > 1:
                    this_partial = copy.deepcopy(combiner)
                    this_partial.keywords['candidate_algorithms'] = candidate_algorithms
                    register_func(this_partial, noise)

        if noise_type == 'bin':
            for noise in oracle_bin_noise_mask:
                mask_combiner(noise)
        elif noise_type == 'logdis':
            for noise in oracle_logdis_noise_mask:
                mask_combiner(noise)
        elif noise_type == 'dis':
            for noise in oracle_dis_noise_mask:
                mask_combiner(noise)
        else:
            raise ValueError('Invalid noise type')

###############################################################
    cache_dict = {}
    caches = []
    funcs = [(outer_key, inner_key, value) for outer_key, inner_dict in func_dict.items() for inner_key, value in inner_dict.items()]

    boost_cls_list = {}
    for name, lst, in boost_preds_dict.items():
        if name in PredictAlgorithmFactory.predictor_evict_dict:
            this_cls = PredictAlgorithmFactory.predictor_evict_dict[name][1]
            boost_cls_list[this_cls] = lst

    with tqdm.tqdm(desc="Init caches for benchmark", total=len(funcs), disable=disable_progress) as pbar:
        def register_cache(pretty_name, noise, cache):
            if pretty_name not in cache_dict:
                cache_dict[pretty_name] = {}
            cache_dict[pretty_name][noise] = cache

        def judge_pred_type(this_partial):
            if hasattr(this_partial, 'keywords'):
                kw = this_partial.keywords
                if 'predictor_type' in kw:
                    predictor_type = kw['predictor_type']
                    pred_cls_type = predictor_type.func if hasattr(predictor_type, 'func') else predictor_type
                    return pred_cls_type
            return None

        for pretty_name, noise, this_partial in funcs:
            if args.oracle:
                if args.boost:
                    # todo
                    cache = Cache(file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
                else:
                    cache = Cache(file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
            elif args.real:
                cache = None
                if args.boost:
                    pred_cls_type = judge_pred_type(this_partial)
                    if pred_cls_type is not None:
                        if pred_cls_type in boost_cls_list:
                            tqdm.tqdm.write(f"Enbale Boost for {pretty_name} --> {pred_cls_type.__name__}")
                            cache = BoostCache(copy.deepcopy(boost_cls_list[pred_cls_type]), file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)

                    if hasattr(this_partial, 'keywords'):
                        kw = this_partial.keywords
                        if 'candidate_algorithms' in kw:
                            algs = kw['candidate_algorithms']
                            for alg in algs:
                                pred_cls_type = judge_pred_type(alg)
                                if pred_cls_type in boost_cls_list:
                                    tqdm.tqdm.write(f"Enbale Boost for {pretty_name} --> {pred_cls_type.__name__}")
                                    cache = BoostCache(copy.deepcopy(boost_cls_list[pred_cls_type]), file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
                                    break
                if cache is None:
                    cache = Cache(file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
            register_cache(pretty_name, noise, cache)
            caches.append(cache)
            pbar.update(1)

    if args.boost and args.oracle:
        num_workers = args.num_workers
        if num_workers == -1:
            num_workers = os.cpu_count()
        
        if not args.quiet:
            print(f'Benchmark: Workers[{num_workers}]')
        with Pool(processes=num_workers) as pool:
            stats = list(tqdm.tqdm(pool.map(process_cache, caches), total=len(caches), disable=disable_progress))
        for i, stat in enumerate(stats):
            caches[i].set_stat(stats[i][0], stats[i][1], stats[i][2])
    else:
        with DataTrace(file_path) as trace:
            with tqdm.tqdm(desc="Producing cache on MemoryTrace", disable=disable_progress) as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    for cache in caches:
                        cache.access(pc, address)
                    pbar.update(1)

    table = PrettyTable() 

    opt_miss = np.inf
    if 'OPT' in cache_dict:
        hit, opt_miss, total, rate = cache_dict['OPT'][0].stat()

    if verbose:
        table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate", "Competitive Ratio"]
        for i, (pretty_name, _, _) in enumerate(funcs):
            hit, miss, total, rate = caches[i].stat()
            table.add_row([pretty_name, hit, miss, total, rate, f"{miss / opt_miss:.3f}"])
        if sorted:
            table.sortby = "Hit Rate"
            table.reversesort = True
    else:
        if noise_type == 'dis':
            table.field_names = ["Name"] + [f'Dis-{noise}' for noise in oracle_dis_noise_mask]
        elif noise_type == 'logdis':
            table.field_names = ["Name"] + [f'LogDis-{noise}' for noise in oracle_logdis_noise_mask]
        elif noise_type == 'bin':
            table.field_names = ["Name"] + [f'Bin-{noise}' for noise in oracle_bin_noise_mask]
        for pretty_name in cache_dict:
            if pretty_name == 'OPT':
                continue
            lst = [pretty_name]
            for noise in cache_dict[pretty_name]:
                _, miss, _, rate = cache_dict[pretty_name][noise].stat()
                lst.append(f'{rate:.3f}/{miss / opt_miss:.3f}')
            if len(lst) == 2:
                lst.extend([lst[-1]] * (len(table.field_names) - 2))
            table.add_row(lst)

    if args.dump_file:
        res_dir = os.path.join(args.output_root_dir, args.dataset, args.model_fraction)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        if args.real:
            with open(os.path.join(res_dir, f"{'_'.join(this_preds)}.csv"), "w", encoding="utf-8") as file:
                file.write(table.get_csv_string())
        else:
            with open(os.path.join(res_dir, f"{args.noise_type}.csv"), "w", encoding="utf-8") as file:
                file.write(table.get_csv_string())
    print(table)
            