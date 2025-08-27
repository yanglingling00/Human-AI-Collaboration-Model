import monitor_process
import control_process
import logging

if __name__ == '__main__':
    set_seed(42)
    currency = 'USDJPY'
    time_frame = '4_Hour'
    human_ability = 0.57
    alpha = 1          # Human-Machine Weight Ratio
    indexs = 5         # Conduct five experiments
    early_epoch = 20

    # Three single models of different dimensions
    short_res = monitor_process.run_monitor_process(6, currency, time_frame, human_ability, early_epoch, alpha, indexs)
    middle_res = monitor_process.run_monitor_process(12, currency, time_frame, human_ability, early_epoch, alpha, indexs)
    long_res = monitor_process.run_monitor_process(24, currency, time_frame, human_ability, early_epoch, alpha, indexs)

    # parameters
    short_seq, short_index = 5,1
    middle_seq, middle_index = 5,1
    long_seq, long_index = 5,1

    # Configure logging to save the results
    logger = logging.getLogger(f'logger_res')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'output.txt')
    logger.addHandler(handler)

    # stage2
    logger.info(control_process.run_control_process(currency, time_frame, short_seq, short_index, middle_seq, middle_index, long_seq,
                                   long_index, human_ability, 200, alpha, indexs))





