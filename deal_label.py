import pandas as pd
label=['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find',
       'service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed',
       'price_level','price_cost_effective','price_discount',
       'environment_decoration','environment_noise','environment_space','environment_cleaness',
       'dish_portion','dish_taste','dish_look','dish_recommendation',
       'others_overall_experience','others_willing_to_consume_again']
########对标签按列存储为TXT########
io=[r"./dataset/sentiment_analysis_trainingset.csv",r"./dataset/sentiment_analysis_testa.csv",r"./dataset/sentiment_analysis_validationset.csv"]
method=['train','test','validation']
for i in range(len(io)):
    true_io=io[i]
    dataframe=pd.read_csv(true_io)
    for label_label in label:
        labelvalue = dataframe[label_label]
        file = open('./dataset/label/%s_label_%s.txt'%(method[i],label_label), 'w',encoding='utf-8')
        for index in range(len(labelvalue)):
            i_value = labelvalue[index]
            file.write(str(i_value)+'\n')
            if index % 2000 == 0:
                print(index)
        file.close()