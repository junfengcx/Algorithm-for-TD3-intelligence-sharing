import numpy as np
import random

class IntelligenceSharingEnv:
    def __init__(self, request_num, provider_num, wireless_band, model_size, train_number, comp_requester, similar_value, number_paras, number_trans_prob, quality_paras, quality_trans_prob, channel_paras, channel_trans_prob, trust_paras, trust_trans_prob, delay_threshold, power_max, comp_max):
        self.request_num = request_num
        self.provider_num = provider_num
        self.wireless_band = wireless_band
        self.model_size = model_size
        self.train_number = train_number
        self.comp_requester = comp_requester
        self.similar_value = similar_value
        self.number_paras = number_paras
        self.number_trans_prob = number_trans_prob
        self.quality_paras = quality_paras
        self.quality_trans_prob = quality_trans_prob
        self.channel_paras = channel_paras
        self.channel_trans_prob = channel_trans_prob
        self.trust_paras = trust_paras
        self.trust_trans_prob = trust_trans_prob
        self.delay_threshold = delay_threshold
        self.power_max = power_max
        self.comp_max = comp_max
        self.data_number = np.zeros(self.provider_num)
        self.data_quality = np.zeros(self.provider_num)
        self.uav_user_channel = np.zeros((self.provider_num, self.request_num))
        self.user_uav_trust = np.zeros((self.request_num, self.provider_num))

    def step(self, request_strategy, power_strategy, computation_strategy):
        request_strategy_new, power_strategy_new, computation_strategy_new = repair_action(request_strategy, power_strategy, computation_strategy, self.power_max, self.comp_max)
        data_number_new, data_quality_new, uav_user_channel_new, user_uav_trust_new = repair_state(self.data_number.copy(), self.data_quality.copy(), self.uav_user_channel.copy(), self.user_uav_trust.copy())
        energy_efficiency = get_energy_efficiency(request_strategy_new, power_strategy_new, computation_strategy_new, self.request_num, self.provider_num, self.wireless_band, uav_user_channel_new, self.model_size, data_number_new, self.train_number, data_quality_new, self.comp_requester, user_uav_trust_new, self.similar_value)
        Delay = get_delay(power_strategy_new, computation_strategy_new, self.request_num, self.provider_num, self.wireless_band, uav_user_channel_new, self.model_size, data_number_new, self.train_number)
        if Delay <= self.delay_threshold:
            reward = energy_efficiency
        else:
            reward = energy_efficiency - 0.5 * (Delay - self.delay_threshold)

        self.data_number = get_data_number_next(self.data_number, self.number_paras, self.number_trans_prob)
        self.data_quality = get_data_quality_next(self.data_quality, self.quality_paras, self.quality_trans_prob)
        self.uav_user_channel = get_uav_user_channel_next(self.uav_user_channel, self.channel_paras, self.channel_trans_prob)
        self.user_uav_trust = get_user_uav_trust_next(self.user_uav_trust, self.trust_paras, self.trust_trans_prob)

        return self.data_number, self.data_quality, self.uav_user_channel, self.user_uav_trust, reward

    def reset(self):
        self.data_number = np.array([self.number_paras[0]] * self.provider_num)
        self.data_quality = np.array([self.quality_paras[0]] * self.provider_num)
        self.uav_user_channel = np.array([[self.channel_paras[0] for j in range(self.request_num)] for i in range(self.provider_num)])
        self.user_uav_trust = np.array([[self.trust_paras[0] for j in range(self.provider_num)] for i in range(self.request_num)])
        return self.data_number, self.data_quality, self.uav_user_channel, self.user_uav_trust

def repair_action(request_strategy, power_strategy, computation_strategy, power_max, comp_max):
    request_strategy_new = request_strategy.copy()
    power_strategy_new = power_strategy.copy() * power_max
    computation_strategy_new = computation_strategy.copy() * comp_max
    request_strategy_new[request_strategy_new < 0] = 0
    request_strategy_new[request_strategy_new > 1] = 1
    power_strategy_new[power_strategy_new <= 0] = 0.001
    power_strategy_new[power_strategy_new > power_max] = power_max
    computation_strategy_new[computation_strategy_new <= 0] = 0.01
    computation_strategy_new[computation_strategy_new > comp_max] = comp_max
    return request_strategy_new, power_strategy_new, computation_strategy_new

def repair_state(data_number, data_quality, uav_user_channel, user_uav_trust):
    data_number_new = data_number.copy()
    data_quality_new = data_quality.copy()
    uav_user_channel_new = uav_user_channel.copy()
    user_uav_trust_new = user_uav_trust.copy()
    shape1 = np.shape(user_uav_trust)
    request_num = shape1[0]
    provider_num = shape1[1]

    for i in range(provider_num):
        if data_number_new[i] == 0.0:
            data_number_new[i] = 100
        elif data_number_new[i] == 0.25:
            data_number_new[i] = 150
        elif data_number_new[i] == 0.5:
            data_number_new[i] = 200
        elif data_number_new[i] == 0.75:
            data_number_new[i] = 250
        elif data_number_new[i] == 1.0:
            data_number_new[i] = 300

        if data_quality_new[i] == 0.0:
            data_quality_new[i] = 0.55
        elif data_quality_new[i] == 0.25:
            data_quality_new[i] = 0.65
        elif data_quality_new[i] == 0.5:
            data_quality_new[i] = 0.75
        elif data_quality_new[i] == 0.75:
            data_quality_new[i] = 0.85
        elif data_quality_new[i] == 1.0:
            data_quality_new[i] = 0.95

    for i in range(request_num):
        for j in range(provider_num):
            if uav_user_channel_new[j, i] == 0.0:
                uav_user_channel_new[j, i] = 5
            elif uav_user_channel_new[j, i] == 0.25:
                uav_user_channel_new[j, i] = 10
            elif uav_user_channel_new[j, i] == 0.5:
                uav_user_channel_new[j, i] = 15
            elif uav_user_channel_new[j, i] == 0.75:
                uav_user_channel_new[j, i] = 20
            elif uav_user_channel_new[j, i] == 1.0:
                uav_user_channel_new[j, i] = 25

            if user_uav_trust_new[i, j] == 0.0:
                user_uav_trust_new[i, j] = 0.35
            elif user_uav_trust_new[i, j] == 0.25:
                user_uav_trust_new[i, j] = 0.5
            elif user_uav_trust_new[i, j] == 0.5:
                user_uav_trust_new[i, j] = 0.65
            elif user_uav_trust_new[i, j] == 0.75:
                user_uav_trust_new[i, j] = 0.8
            elif user_uav_trust_new[i, j] == 1.0:
                user_uav_trust_new[i, j] = 0.95

    return data_number_new, data_quality_new, uav_user_channel_new, user_uav_trust_new

def get_energy_efficiency(request_strategy, power_strategy, computation_strategy, request_num, provider_num, wireless_band, channel_gain, model_size, data_number, train_number, data_quality, comp_requester, trust_value, similar_value):
    provider_request_rate = RadioRateGenerator(request_num, provider_num, power_strategy, wireless_band, channel_gain)
    totalCommEnergy = CommEnergyGenerator(request_num, provider_num, power_strategy, request_strategy, model_size, provider_request_rate)
    totalCompEnergy = CompEnergyGenerator(provider_num, data_number, computation_strategy, train_number)
    averge_energy = (totalCommEnergy + totalCompEnergy) / provider_num
    modelAccuracy = modelAccuracyEachProviderGenerator(provider_num, data_quality, data_number, computation_strategy, train_number)
    total_model_performance = modelAccuracyTotalRequestGenerator(request_num, provider_num, comp_requester, modelAccuracy, request_strategy, trust_value, similar_value)
    average_model_performance = total_model_performance / request_num
    energy_efficiency = (10 ** 4) * average_model_performance / averge_energy
    return energy_efficiency

def get_delay(power_strategy, computation_strategy, request_num, provider_num, wireless_band, channel_gain, model_size, data_number, train_number):
    provider_request_rate = RadioRateGenerator(request_num, provider_num, power_strategy, wireless_band, channel_gain)
    CommDelay_each_provider = CommDelayGenerator(request_num, provider_num, model_size, provider_request_rate)
    CompDelay_each_provider = CompDelayGenerator(provider_num, data_number, computation_strategy, train_number)
    Delay_each_provider = CommDelay_each_provider + CompDelay_each_provider
    Delay = Delay_each_provider.max()
    return Delay

def RadioRateGenerator(request_num, provider_num, power_strategy, wireless_band, channel_gain):
    provider_request_rate = np.zeros((provider_num, request_num))
    each_user_band = 180
    for i in range(provider_num):
        for j in range(request_num):
            provider_request_rate[i, j] = each_user_band * np.log2(1 + power_strategy[i] * channel_gain[i, j])
    return provider_request_rate

def CommEnergyGenerator(request_num, provider_num, power_strategy, request_strategy, model_size, provider_request_rate):
    totalCommEnergy = 0
    for i in range(provider_num):
        for j in range(request_num):
            totalCommEnergy = totalCommEnergy + (request_strategy[j, i] * power_strategy[i] * model_size) / provider_request_rate[i, j]
    return totalCommEnergy

def CommDelayGenerator(request_num, provider_num, model_size, provider_request_rate):
    CommDelay_each_provider_requester = np.zeros((provider_num, request_num))
    CommDelay_each_provider = np.zeros(provider_num)
    for i in range(provider_num):
        for j in range(request_num):
            CommDelay_each_provider_requester[i, j] = model_size / provider_request_rate[i, j]
    for i in range(provider_num):
        CommDelay_each_provider[i] = CommDelay_each_provider_requester[i].max()
    return CommDelay_each_provider

def CompEnergyGenerator(provider_num, data_number, computation_strategy, train_number):
    per_energy = 1
    comp_per_data = 0.25
    totalCompEnergy = 0
    for i in range(provider_num):
        totalCompEnergy = totalCompEnergy + per_energy * train_number * comp_per_data * data_number[i] * computation_strategy[i] * computation_strategy[i]
    return totalCompEnergy

def CompDelayGenerator(provider_num, data_number, computation_strategy, train_number):
    CompDelay_each_provider = np.zeros(provider_num)
    comp_per_data = 0.25
    for i in range(provider_num):
        CompDelay_each_provider[i] = train_number * comp_per_data * data_number[i] / computation_strategy[i]
    return CompDelay_each_provider

def modelAccuracyEachProviderGenerator(provider_num, data_quality, data_number, computation_strategy, train_number):
    w_lc = 0.1
    alerfa = 0.5
    vei = 0.5
    modelAccuracy = np.array([0.0 for i in range(provider_num)])
    for i in range(provider_num):
        temp = data_quality[i] * data_number[i] * (train_number ** alerfa)
        temp1 = (- w_lc) * computation_strategy[i] * (temp ** vei)
        modelAccuracy[i] = 1 - np.exp(temp1)
    return modelAccuracy

def modelAccuracyTotalRequestGenerator(request_num, provider_num, comp_requester, modelAccuracy, request_strategy, trust_value, similar_value):
    model_performance = np.array([0.0 for i in range(request_num)])
    total_model_performance = 0
    w_ag = 3
    model_accuracy_requester = np.zeros(request_num)
    model_accuracy_sum_up = np.zeros(request_num)
    model_accuracy_sum_down = np.zeros(request_num)
    for i in range(request_num):
        for j in range(provider_num):
            model_accuracy_sum_up[i] = model_accuracy_sum_up[i] + request_strategy[i, j] * similar_value[i, j] * trust_value[i, j] * modelAccuracy[j]
    for i in range(request_num):
        for j in range(provider_num):
            model_accuracy_sum_down[i] = model_accuracy_sum_down[i] + request_strategy[i, j] * similar_value[i, j] * trust_value[i, j]
    for i in range(request_num):
        model_accuracy_requester[i] = model_accuracy_sum_up[i] / model_accuracy_sum_down[i]

    for i in range(request_num):
        model_performance[i] = 1 - np.exp((- w_ag) * comp_requester[i] * model_accuracy_requester[i])
    for i in range(request_num):
        total_model_performance = total_model_performance + model_performance[i]
    return total_model_performance

def get_data_number_next(data_number, number_paras, number_trans_prob):
    shape1 = np.shape(data_number)
    provider_num = shape1[0]
    shape2 = np.shape(number_paras)
    number_num = shape2[0]
    data_number_next = np.zeros(provider_num)
    for i in range(provider_num):
        temp = np.random.choice(np.arange(number_num),
                                p=number_trans_prob[np.where(number_paras == data_number[i])[0][0]])
        data_number_next[i] = number_paras[temp]
    return data_number_next

def get_data_quality_next(data_quality, quality_paras, quality_trans_prob):
    shape1 = np.shape(data_quality)
    provider_num = shape1[0]
    shape2 = np.shape(quality_paras)
    quality_num = shape2[0]
    data_quality_next = np.zeros(provider_num)
    for i in range(provider_num):
        temp = np.random.choice(np.arange(quality_num),
                                p=quality_trans_prob[np.where(quality_paras == data_quality[i])[0][0]])
        data_quality_next[i] = quality_paras[temp]
    return data_quality_next

def get_uav_user_channel_next(uav_user_channel, channel_paras, channel_trans_prob):
    shape1 = np.shape(uav_user_channel)
    provider_num = shape1[0]
    request_num = shape1[1]
    shape2 = np.shape(channel_paras)
    channel_num = shape2[0]
    uav_user_channel_next = np.zeros((provider_num, request_num))
    for i in range(provider_num):
        for j in range(request_num):
            temp = np.random.choice(np.arange(channel_num), p=channel_trans_prob[np.where(channel_paras == uav_user_channel[i, j])[0][0]])
            uav_user_channel_next[i, j] = channel_paras[temp]
    return uav_user_channel_next

def get_user_uav_trust_next(user_uav_trust, trust_paras, trust_trans_prob):
    shape1 = np.shape(user_uav_trust)
    request_num = shape1[0]
    provider_num = shape1[1]
    shape2 = np.shape(trust_paras)
    trust_num = shape2[0]
    user_uav_trust_next = np.zeros((request_num, provider_num))
    for i in range(request_num):
        for j in range(provider_num):
            temp = np.random.choice(np.arange(trust_num), p=trust_trans_prob[np.where(trust_paras == user_uav_trust[i, j])[0][0]])
            user_uav_trust_next[i, j] = trust_paras[temp]
    return user_uav_trust_next

def get_random_request_strategy(request_num, provider_num):
    random_request_strategy = np.random.rand(request_num, provider_num)
    return random_request_strategy

def get_random_power_strategy(provider_num, power_max):
    random_power_strategy = np.random.uniform(low=0.001, high=power_max, size=provider_num)
    return random_power_strategy

def get_random_computation_strategy(provider_num, comp_max):
    random_computation_strategy = np.random.uniform(low=0.01, high=comp_max, size=provider_num)
    return random_computation_strategy

def get_three_strategy(actions_value, request_num, provider_num):
    request_strategy = np.zeros((request_num, provider_num))
    power_strategy = np.zeros(provider_num)
    computation_strategy = np.zeros(provider_num)
    power_strategy = actions_value[(request_num * provider_num):((request_num+1) * provider_num)]
    computation_strategy = actions_value[((request_num+1) * provider_num):((request_num+2) * provider_num)]
    for i in range(request_num):
        request_strategy[i] = actions_value[(i * provider_num):(i * provider_num + provider_num)]
    request_strategy[request_strategy < 0] = 0
    request_strategy[request_strategy > 1] = 1
    power_strategy[power_strategy < 0] = 0
    power_strategy[power_strategy > 1] = 1
    computation_strategy[computation_strategy < 0] = 0
    computation_strategy[computation_strategy > 1] = 1
    return request_strategy, power_strategy, computation_strategy

def get_two_strategy(actions_value, request_num, provider_num):
    request_strategy = np.zeros((request_num, provider_num))
    power_strategy = np.zeros(provider_num)
    power_strategy = actions_value[(request_num * provider_num):((request_num+1) * provider_num)]
    for i in range(request_num):
        request_strategy[i] = actions_value[(i * provider_num):(i * provider_num + provider_num)]
    request_strategy[request_strategy < 0] = 0
    request_strategy[request_strategy > 1] = 1
    power_strategy[power_strategy < 0] = 0
    power_strategy[power_strategy > 1] = 1
    return request_strategy, power_strategy

def get_one_strategy(actions_value):
    computation_strategy = actions_value
    computation_strategy[computation_strategy < 0] = 0
    computation_strategy[computation_strategy > 1] = 1
    return computation_strategy







