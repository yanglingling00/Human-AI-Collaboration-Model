import singel_model
import stage2
import logging

if __name__ == '__main__':
    currency = 'USDJPY'
    time_frame = '4_Hour'
    human_ability = 0.57
    alpha = 1          # 人机权重比
    indexs = 5         # 5次实验均值
    early_epoch = 20

    # 三个不同维度的单个模型
    short_res = singel_model.run_single_model(6, currency, time_frame, human_ability, early_epoch, alpha, indexs)
    middle_res = singel_model.run_single_model(12, currency, time_frame, human_ability, early_epoch, alpha, indexs)
    long_res = singel_model.run_single_model(24, currency, time_frame, human_ability, early_epoch, alpha, indexs)

    # 选择参数
    short_seq, short_index = 5,1
    middle_seq, middle_index = 5,1
    long_seq, long_index = 5,1

    # 配置logging 保存结果
    logger = logging.getLogger(f'logger_res')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'output.txt')
    logger.addHandler(handler)

    # stage2
    # logger.info(stage2.run_stage_2(currency, time_frame, short_seq, short_index, middle_seq, middle_index, long_seq,
    #                                long_index, human_ability, 200, alpha, indexs))

    res, sys_acc = stage2.run_stage_2(currency, time_frame, short_seq, short_index, middle_seq, middle_index, long_seq,
                                   long_index, human_ability, 200, alpha, indexs)

    logger.info(res)