import math
import socket
import time
import pandas as pd
import numpy as np
import copy


from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case
import result_value.value as gl

# Configurations are in a separate config.py file
from config import *

model = get_model(model_name)#当前的模型，不同的名称对应不同的模型

##神经网络需要创建计算流图
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

#time_gen？
if time_gen is not None:
    use_fixed_averaging_slots = True
else:
    use_fixed_averaging_slots = False

#一次性读取所有数据
if batch_size < total_data:   # Read all data once when using stochastic gradient descent
    train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)

    # This function takes a long time to complete,
    # putting it outside of the sim loop because there is no randomness in the current way of computing the indices
    #获得需要的各个边缘节点的数据样本
    indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

#建立网络通信
listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
client_sock_all=[]

# Establish connections to each client, up to n_nodes clients
while len(client_sock_all) < n_nodes:
    listening_sock.listen(5)
    print("-----------------------------------Waiting for incoming connections-----------------------------------")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip,port))
    # print(client_sock)
    client_sock_all.append(client_sock)


#创建结果统计对象
if single_run:
    stat = CollectStatistics(results_file_name=single_run_results_file_path, is_single_run=True)
else:
    stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)

for sim in sim_runs:

    if batch_size >= total_data:  # Read data again for different sim. round
        train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

        # This function takes a long time to complete,
        indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)
    else:
        print("batch_size,total_data",batch_size,total_data)

    #不同分布的数据集
    #df = pd.DataFrame(columns=["sim", "case", "loss", "accuracy"])
    df = pd.DataFrame(columns=["case", "tau_setup", "tau_config", "accuracy"])
    pf = pd.DataFrame(columns=["case", "aggregation_times", "tau_config"])
    pss=[]
    for case in case_range:
        #本地运行频率 tau_setup_all = [-1, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100]
        aggregation_times=0
        tau_setup_all=[-1]
        for tau_setup in tau_setup_all:
            #tau_setup设置tau，dim_w初始化的权重的维度784
            #stat统计对象的初始化
            stat.init_stat_new_global_round()
            dim_w = model.get_weight_dimension(train_image, train_label)
            #对模型的权重进行初始化
            w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
            #设定全局模型参数为初始值
            w_global = w_global_init
            #全局最小的损失值
            w_global_min_loss = None
            loss_min = np.inf#正无穷大的浮点表示
            #前一次全局的最小损失值
            prev_loss_is_min = False
            #当本地更新次数小于0的时候进行动态调整，设置本地更新次数为1
            is_adapt_local = True
            tau_config = 1
            ##控制算法的对象实例化
            control_alg = ControlAlgAdaptiveTauServer(is_adapt_local, dim_w, client_sock_all, n_nodes,
                                                              control_param_phi, moving_average_holding_param)
            ###每个节点都要进行处理
            for n in range(0, n_nodes):
                ##获取对应的节点数据集
                indices_this_node = indices_each_node_case[case][n]
                msg = ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset,
                       num_iterations_with_same_minibatch_for_tau_equals_one, step_size,
                       batch_size, total_data, control_alg, indices_this_node, read_all_data_for_stochastic,
                       use_min_loss, sim]
                ##发送初始数据
                send_msg(client_sock_all[n], msg)
            print('--------------------------------------All clients MSG_INIT_SERVER_TO_CLIENT connected----------------------------------')
            #接收消息，知道各个节点都已经收到了相对应的数据
            for n in range(0, n_nodes):
                recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')
                print("-----------------",case,tau_setup)
                #recv_msg(client_sock_all[n], 'all_client_receive_init_pramras')
            ##开始进行边缘节点的协作训练
            print('Start learning--------------------------------------------------')
            #time_global_aggregation_all开始记录所有的数据
            time_global_aggregation_all = None
            #总的时间
            total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
            #重新估计的时间
            total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,
            # using predefined values when use_fixed_averaging_slots = true
            it_each_local = None
            it_each_global = None
            ##是否是最后一次训练
            is_last_round = False
            is_eval_only = False
            tau_new_resume = None
            # Loop for multiple rounds of local iterations + global aggregation
            ###正式开始进行训练---------------------------------------------------
            # w_global2 = copy.deepcopy(w_global)
            # w_global3 = copy.deepcopy(w_global)
            print('--------------------------start traning -------------------------------------------------')
            while True:
                # 当前运行中的数据存储，方便对实验结果进行分析
                dflist = []
                gl.COM_TIMES=gl.COM_TIMES+1
                dflist.append(gl.COM_TIMES)
                #print('current tau config:',gl.COM_TIMES, tau_config)
                dflist.append(tau_config)
                pflist=[]
                pflist.append(aggregation_times)
                pflist.append(tau_setup)
                pflist.append(tau_config)
                pss.append(pflist)

                time_total_all_start = time.time()
                #将基本的信息传送给各个节点
                for n in range(0, n_nodes):
                    msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min]
                    #msg = ['model_pramas_to_client', w_global2, tau_config, is_last_round, prev_loss_is_min]
                    send_msg(client_sock_all[n], msg)

                w_global_prev = w_global
                #新的全局模型参数
                w_global = np.zeros(dim_w)
                loss_last_global = 0.0
                loss_w_prev_min_loss = 0.0
                received_loss_local_w_prev_min_loss = False
                data_size_total = 0
                time_all_local_all = 0
                data_size_local_all = []
                tau_actual = 0
                #新增，对每一个节点进行操作
                loss_list=[]
                w_local_list=[]
                print('--------------------------------Waiting for local iteration at client------------------------------')
                for n in range(0, n_nodes):
                    msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                    #msg = recv_msg(client_sock_all[n], 'client_local_updates_paramas_to_server')
                    # ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
                    # loss_last_global, loss_w_prev_min_loss]
                    w_local = msg[1]#本地的模型权重
                    time_all_local = msg[2]#本地执行的时间消耗
                    #tau_actual初始值，msg[3]评估出来的tau(tau_actual)
                    tau_actual = max(tau_actual, msg[3])  # Take max of tau because we wait for the slowest node
                    data_size_local = msg[4]#本地的数据量
                    loss_local_last_global = msg[5]#最近一次的本地模型的损失
                    loss_local_w_prev_min_loss = msg[6]#最小的本地模型的损失值
                    # w_global2 = w_global2 + w_local * (1-loss_local_last_global)
                    # w_global = w_global+ w_local * (1 - loss_local_last_global)
                    data_size_local_all.append(data_size_local)
                    data_size_total += data_size_local
                    #取几个节点的消耗时间的最大值
                    time_all_local_all = max(time_all_local_all, time_all_local)   #Use max. time to take into account the slowest node
                    w_global += w_local * data_size_local
                    #计算最近一次全局损失（F(w)=all(Fi(w)*data_size_local/data_size_total）)
                    loss_last_global += loss_local_last_global * data_size_local
                    #计算目前的最小损失，loss_local_w_prev_min_loss是client本地的最小损失
                    if loss_local_w_prev_min_loss is not None:
                        loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
                        received_loss_local_w_prev_min_loss = True
                    # # 新增
                    # #print("--------------------", loss_local_last_global, loss_local_w_prev_min_loss)
                    # loss_list.append(loss_local_last_global)
                    # w_local_list.append(w_local)
                w_global /= data_size_total
#此时的w_global已经是一个全局模型的权重参数
                # sum=0
                # for i in range(len(loss_list)):
                #     sum=sum+loss_list[i]
                #     # print(np.log(loss_list[i]))
                #    # print(w_local_list[i]*np.log(loss_list[i]))
                #     new_wlocal=w_local_list[i]*np.log(loss_list[i])
                #     # w_global=w_global+w_local_list[i]*np.log(loss_list[i])


                #print(w_global2)
                # for i in range(len(loss_list)):
                #     rate=loss_list[i]/sum
                #     w_global2 +=w_local_list[i]*(1-rate)
                #
                #w_global=w_global2
                #w_global = copy.deepcopy(w_global2)

                use_w_global_prev_due_to_nan = False
#处理w_global is nan 的情况,则考虑用t-1时刻的全局模型
                if True in np.isnan(w_global):
                    print('************************************ w_global is NaN, using previous value ******************************************')
                    w_global = w_global_prev   # If current w_global contains NaN value, use previous w_global
                    #是否使用t-1时刻的全局参数来替代nan(当t时刻参数是nan时)
                    use_w_global_prev_due_to_nan = True
                #use_min_loss=True
#计算t-1时刻的全局损失loss_last_global，如果使用最小损失use_min_loss=true,
 # 先计算t-1时刻的global loss
                # received_loss_local_w_prev_min_loss默认Flase,
                # 当得到client节点传送各自的损失时received_loss_local_w_prev_min_loss=True
                loss_last_global /= data_size_total


                if received_loss_local_w_prev_min_loss:
                        # 计算前t-1时刻对应的全局损失loss_min,loss_w_prev_min_loss
                    loss_w_prev_min_loss /= data_size_total
                    loss_min = loss_w_prev_min_loss
                     #判定t-1时刻损失和最小损失之间的关系对loss_min，w_global_min_loss，prev_loss_is_min进行更新
                if loss_last_global < loss_min:
                    loss_min = loss_last_global
                    w_global_min_loss = w_global_prev
                    prev_loss_is_min = True
                else:
                    prev_loss_is_min = False
                # If use_w_global_prev_due_to_nan, then use tau = 1 for next round
                #当我们不适用t-1时刻的权重参数时，自适应算法计算相关参数
                #计算 tau_new
                if not use_w_global_prev_due_to_nan:
                # Only update tau if use_w_global_prev_due_to_nan is False
                    tau_new = control_alg.compute_new_tau(data_size_local_all, data_size_total,
                                                              it_each_local, it_each_global, max_time,
                                                              step_size, tau_config, use_min_loss)
                else:
                    if tau_new_resume is None:
                        tau_new_resume = tau_config
                    tau_new = 1

                # Calculate time,计算资源消耗
                time_total_all_end = time.time()
                time_total_all = time_total_all_end - time_total_all_start
                time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)
                local_time=time_all_local_all / tau_actual

                # print('Time for one local iteration:', local_time)
                # print('Time for global averaging:', time_global_aggregation_all)

                #参数存放
                dflist.append(time_total_all)
                dflist.append(local_time)
                dflist.append(time_global_aggregation_all)
                dflist.append(loss_last_global)
                gl.DF.loc[len(gl.DF)+ 1] = dflist

               #用time_gen，tau_actual，time_global_aggregation_all，time_all_local_all计算 it_each_local，it_each_global时间
                if use_fixed_averaging_slots:
                    if isinstance(time_gen, (list,)):
                        t_g = time_gen[case]
                    else:
                        t_g = time_gen
                    it_each_local = max(0.00000001, np.sum(t_g.get_local(tau_actual)) / tau_actual)
                    it_each_global = t_g.get_global(1)[0]
                else:
                    it_each_local = max(0.00000001, time_all_local_all / tau_actual)
                    it_each_global = time_global_aggregation_all

                #Compute number of iterations is current slot
                total_time_recomputed += it_each_local * tau_actual + it_each_global
                #Compute time in current slot
                total_time += time_total_all
                stat.collect_stat_end_local_round(case, tau_actual, it_each_local, it_each_global, control_alg, model,
                                                  train_image, train_label, test_image, test_label, w_global,
                                                  total_time_recomputed)

                # Check remaining resource budget (use a smaller tau if the remaining time is not sufficient)
                is_last_round_tmp = False
                tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * (tau_new + 1) + it_each_global * 2


                #资源消耗统计，如果资源没有消耗完则tau_config = tau_new，否则设置最后一个is_last_round_tmp
                if tmp_time_for_executing_remaining < max_time:
                    tau_config = tau_new
                else:
                    tau_config = int((max_time - total_time_recomputed - 2 * it_each_global - it_each_local) / it_each_local)
                    if tau_config < 1:
                        tau_config = 1
                    elif tau_config > tau_new:
                        tau_config = tau_new
                    is_last_round_tmp = True

                if is_last_round:
                    stat.collect_paramas(sim, case, tau_setup, total_time, model, train_image, train_label,
                                         test_image, test_label, w_global, total_time_recomputed)
                    break

                if is_eval_only:
                    tau_config = 1
                    is_last_round = True

                if is_last_round_tmp:
                    is_eval_only = True


            aggregation_times=aggregation_times+1
            #w_eval 是最终的模型权重数据，也就是server上的模型
            # w_global_min_loss，最小损失对应的模型权重
            w_eval = w_global_min_loss
            # w_eval = copy.deepcopy(w_global2)
            # w_global_prev=copy.deepcopy(w_global2)
            stat.collect_stat_end_global_round(sim, case, tau_setup, total_time, model, train_image, train_label,
                                               test_image, test_label, w_eval, total_time_recomputed)
        for i in pss:
            #print(i)
            pf.loc[len(pf) + 1] =i
        pf.to_csv(gl.PATH + "parama.csv", mode='a')


        # loss_final = model.loss(train_image, train_label, w_global)
        # accuracy_final = model.accuracy(test_image, test_label, w_global)
        # re_list=[sim,case,loss_final,accuracy_final]
        # print(re_list)

        #print(df)
        #df.loc[len(df) + 1] = re_list

    # df.to_csv(gl.PATH + "accracy.csv", mode='a')
    gl.DF.to_csv(gl.PATH+"result.csv")