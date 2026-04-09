import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class GeneticAlgorithmBP:
    def __init__(self, num_inputs, num_hidden, num_outputs, population_size, mutation_rate, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate

        self.population = []
        self.population_size = population_size
        # 通过变异操作能够增大样本输入空间，类似于学习率，此处也以参数形式传入
        self.mutation_rate = mutation_rate
        # 种群中各个体的适合度，fitness_values的大小与种群中个体数相同
        self.fitness_values = np.zeros(population_size)
        # 后续操作中需要频繁使用随机取数的操作，所有在类中内置一个随机数生成器
        self.random_state = check_random_state(None)  # 设置随机数种子

    def data_preprocessing(self, x):
        # 数据预处理：对输入数据进行标准化处理
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        # 防止除以零错误，如果std接近0，则将其设为1
        std[std == 0] = 1
        return (x - mean) / std, mean, std

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 初始化种群，也是初始化神经网络权重的过程，将某次随机生成的权重当做种群中的一个对象
    def initialize_population(self):
        # 短横杠表示临时参数，和占位符一个作用
        for _ in range(self.population_size):
            input_hidden_weights = np.random.uniform(low=-1, high=1, size=(self.num_inputs, self.num_hidden))
            hidden_output_weights = np.random.uniform(low=-1, high=1, size=(self.num_hidden, self.num_outputs))
            self.population.append((input_hidden_weights, hidden_output_weights))

    # 适合度函数
    def evaluate_fitness(self, x, y):
        # 计算中权重每个对象（权重）的适合度，并将适合度保存到fitness_values中
        for i in range(self.population_size):
            # 此处为前向传播，计算中间层和输出层的激活值
            input_hidden_weights, hidden_output_weights = self.population[i]
            hidden_activations = self.sigmoid(np.dot(x, input_hidden_weights))
            output_activations = np.dot(hidden_activations, hidden_output_weights)
            # 以方差作为损失函数，将损失函数的倒数作为适合度，损失函数越小，适合度函数值越大
            error = np.mean((output_activations - y) ** 2)
            self.fitness_values[i] = 1 / (error + 1e-8)

    def selection(self):
        # 计算总适合度，并将适合度转换为被选中的概率
        total_fitness = np.sum(self.fitness_values)
        selection_probs = self.fitness_values / total_fitness
        # 从种群中随机选择一些对象（依据概率），并将其保存到selected_population中
        selected_indices = self.random_state.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        selected_population = []
        for idx in selected_indices:
            selected_population.append(self.population[idx])

        self.population = selected_population

    # 交叉（繁衍行为的主要操作）
    def crossover(self):
        offspring_population = []
        # 计算总适合度，并将适合度转换为被选中的概率，因为我们有理由让适合度高的个体参与到繁衍的过程中
        total_fitness = np.sum(self.fitness_values)
        selection_probs = self.fitness_values / total_fitness

        for _ in range(self.population_size):
            # parent是input_hidden_weights、hidden_output_weights组合而成的元组
            parent1_indices = self.random_state.choice(np.arange(self.population_size),p=selection_probs)
            parent2_indices = self.random_state.choice(np.arange(self.population_size),p=selection_probs)
            input_hidden_weights1, hidden_output_weights1 = self.population[parent1_indices]
            input_hidden_weights2, hidden_output_weights2 = self.population[parent2_indices]
            # 对w1进行交叉操作，交叉点随机（把input1前半部分和Input2后半部分拼接）
            crossover_point = self.random_state.randint(0, self.num_hidden)
            input_hidden_weights = np.concatenate((input_hidden_weights1[:, :crossover_point], input_hidden_weights2[:, crossover_point:]), axis=1)
            # 对w2进行交叉操作，交叉点随机
            crossover_point = self.random_state.randint(0, self.num_outputs)
            hidden_output_weights = np.concatenate((hidden_output_weights1[:, :crossover_point], hidden_output_weights2[:, crossover_point:]), axis=1)
            # 将繁衍产生的对象放入新的种群，直至生成population_size个对象
            offspring_population.append((input_hidden_weights, hidden_output_weights))

        self.population = offspring_population

    # 变异
    def mutation(self):
        for i in range(self.population_size):
            input_hidden_weights, hidden_output_weights = self.population[i]

            if self.random_state.rand() < self.mutation_rate:
                input_hidden_weights += np.random.uniform(low=-0.1, high=0.1, size=(self.num_inputs, self.num_hidden))

            if self.random_state.rand() < self.mutation_rate:
                hidden_output_weights += np.random.uniform(low=-0.1, high=0.1, size=(self.num_hidden, self.num_outputs))

            self.population[i] = (input_hidden_weights, hidden_output_weights)

    def plot_loss(self, fitness_history, validation_losses):
        # 绘制损失函数
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(fitness_history) + 1), fitness_history, label='train')
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, c='orange', label='validation')
        plt.title('Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_predictions(self, x, y, p):
        # 使用索引作为x轴坐标
        X = np.arange(len(x))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # 绘制训练数据点
        ax.plot(X, y[:, 0], label='Actual', marker='o', alpha=0.4)
        # 绘制预测结果点，使用不同的颜色或标记
        # ax.scatter(X, predictions, c='r', cmap='autumn',label='Predicted', marker='x')
        ax.plot(X, p, c='r', label='Predicted', marker='x')
        # 设置x轴刻度为整数
        ax.xaxis.set_major_locator(MultipleLocator(1))  # 设置刻度间隔为1
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # 设置刻度标签为整数格式
        # ax.set_xlim(left=0, right=len(x) - 1)  # 确保x轴从0开始，到数据长度-1结束
        ax.set_title('Training Data vs Predicted Labels')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        plt.show()

    def performance_measures(self, y, p):
        # y为真实值，p为模型预测值
        # 计算MAE、MSE、RMSE
        MAE = np.mean(np.abs(y - p))
        MSE = np.mean((y - p) ** 2)
        RMSE = np.sqrt(MSE)
        R2 = r2_score(y, p)
        # 输出
        print("MAE = " + str(MAE))
        print("MSE = " + str(MSE))
        print("RMSE = " + str(RMSE))
        print("R2 = " + str(R2))

    def evaluate_on_validation_set(self, x_val, y_val):
        # 使用当前最佳权重进行预测
        best_input_hidden_weights, best_hidden_output_weights = self.population[np.argmax(self.fitness_values)]
        hidden_activations_val = self.sigmoid(np.dot(x_val, best_input_hidden_weights))
        output_activations_val = np.dot(hidden_activations_val, best_hidden_output_weights)
        val_loss = np.mean((output_activations_val - y_val) ** 2)

        return val_loss

    def train(self, x, y, num_generations):
        self.initialize_population()
        fitness_history = []  # 用于记录每一代的最佳适应度
        validation_losses = []  # 记录每一代在验证集上的损失
        no_improvement_count = 0  # 连续未改进的代数计数器
        best_validation_loss = float('inf')  # 初始化最佳验证损失为正无穷大
        patience = 200   # 没有观察到验证损失显著降低就停止训练的世代数阈值
        improvement_threshold = 0.001   # 损失必须至少减少0.1%才能被视为改进

        for generation in range(num_generations):
            self.evaluate_fitness(x, y)
            self.selection()
            self.evaluate_fitness(x, y)  # 重新评估适应度
            self.crossover()
            self.mutation()

            # 使用验证集评估模型
            validation_loss = self.evaluate_on_validation_set(x_val, y_val)
            validation_losses.append(validation_loss)

            # 加入早停策略
            # 检查是否有改进，并更新最佳验证损失及计数器
            if validation_loss < best_validation_loss * (1 - improvement_threshold):
                best_validation_loss = validation_loss
                no_improvement_count = 0  # 重置计数器，因为有改进
            else:
                no_improvement_count += 1

            # 提前终止检查
            if no_improvement_count >= patience:
                print(f"Early stopping triggered at generation {generation} due to no improvement.")
                break  # 跳出循环，提前终止训练

            # 记录每一代的最佳适应度
            best_fitness = np.max(self.fitness_values)
            loss = 1 / best_fitness
            fitness_history.append(loss)
            # 输出迭代信息
            if generation % 100 == 0:
                print(str(generation) + "/" + str(num_generations) + "......loss=" + str(loss) + "  val_loss=" +
                      str(validation_loss))
        # 绘制损失函数曲线
        self.plot_loss(fitness_history, validation_losses)
        # 返回最佳权重和验证集上的最佳性能指标
        best_weights = self.population[np.argmax(self.fitness_values)]
        input_hidden_weights, hidden_output_weights = best_weights

        return input_hidden_weights, hidden_output_weights, min(validation_losses)


if __name__ == '__main__':
    # 设置神经网络和遗传算法参数
    num_inputs = 7      # 输入层
    num_hidden = 9     # 隐含层
    num_outputs = 1     # 输出层
    population_size = 200       # 种群大小
    mutation_rate = 0.01        # 变异概率
    learning_rate = 0.05         # 学习率
    num_generations = 1000     # 迭代次数

    # 创建GA-BP神经网络对象
    ga_bp = GeneticAlgorithmBP(num_inputs, num_hidden, num_outputs, population_size, mutation_rate, learning_rate)

    # 创建训练数据集(并进行标准化)并划分数据集（训练集（60%）、验证集（20%）和测试集（20%））
    df = pd.read_excel("广州.xlsx", sheet_name="Sheet1")
    # 将DataFrame转换为numpy数组
    x_all = df.iloc[:, 2:9].values
    y_all = df.iloc[:, 9].values.reshape(-1, 1)

    # 分割数据集
    x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.11,
                                                      random_state=42)  # 这里确保验证集占剩余数据的1/4，即总体的1/5
    # print(len(x_train))
    # print(len(x_val))
    # print(len(x_test))

    # 对训练集和验证集进行预处理
    x_train, _, _ = ga_bp.data_preprocessing(x_train)
    y_train, _, _ = ga_bp.data_preprocessing(y_train)
    x_val, _, _ = ga_bp.data_preprocessing(x_val)
    y_val, _, _ = ga_bp.data_preprocessing(y_val)
    # 对测试集进行预处理并保存标准差与均值用于反标准化
    x_test, _, _ = ga_bp.data_preprocessing(x_test)
    y_test, y_test_mean, y_test_std = ga_bp.data_preprocessing(y_test)

    # 读取input_hidden_weights.npy文件
    if os.path.exists('input_hidden_weights.npy'):
        # 读取权重文件
        input_hidden_weights = np.load('input_hidden_weights.npy')
        hidden_output_weights = np.load('hidden_output_weights.npy')
    else:
        # 训练神经网络
        input_hidden_weights, hidden_output_weights, val_losses = ga_bp.train(x_train, y_train, num_generations)
        print("验证集最小的MSE=" + str(val_losses))
        # 保存权重
        np.save('input_hidden_weights.npy', input_hidden_weights)
        np.save('hidden_output_weights.npy', hidden_output_weights)

    # 使用训练好的权重对测试集进行预测
    hidden_activations = ga_bp.sigmoid(np.dot(x_test, input_hidden_weights))
    output_activations = np.dot(hidden_activations, hidden_output_weights)
    # predictions = np.round(output_activations)
    y_test_predictions = output_activations
    # print("预测结果：")
    # print(y_test_predictions)
    # print(y_test)

    # 计算MAE、MSE和RMSE结果
    print("在测试集上模型评价指标结果如下所示：")
    ga_bp.performance_measures(y_test, y_test_predictions)

    # 反标准化
    y_test = y_test * y_test_std + y_test_mean
    y_test_predictions = y_test_predictions * y_test_std + y_test_mean

    # 绘制训练数据与预测结果的对比图
    ga_bp.plot_predictions(x_test, y_test, y_test_predictions)
