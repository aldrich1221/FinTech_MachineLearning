noHu = []
noTrade = []
trade = []
cluster = ["AA0329","AA0340","AA0546","AA0557","AA0947","AA1529","AA2346","AA2505","AA2969","AA3006","AA3177","AA3180","AA3426","AA3428","AA3509","AA3521","AA3532","AA3548","AA3558","AA3559","AA3567","AA3569","AA3571","AA3580","AA3584","AA3586","AA3605","AA3621","AA3622","AA3631","AA3633","AA3674","AA3678","AA3685","AA3735","AA3741","AA3801","AA4254","AA4259","AA4642","AA5153","AA5297","AA6007","AA7030","AA7384","AA7485","AA7534","AA8459","AA9022","AA9107","AA9122","AA9175","AA9802"]
keyword = ['AA3404', 'AA2888', 'AA3497', 'AA6342', 'AA3522', 'AA2827', 'AA3359', 'AA3521', 'AA8303', 'AA3312', 'AA3525', 'AA2844', 'AA8857', 'AA2764', 'AA2285', 'AA3139', 'AA2885', 'AA9122', 'AA5576', 'AA0809', 'AA3121', 'AA2874', 'AA3586', 'AA3622', 'AA2082', 'AA0340', 'AA2833', 'AA2898', 'AA7485', 'AA2849', 'AA3002', 'AA3732', 'AA3801', 'AA1049', 'AA5706', 'AA3078', 'AA3533', 'AA3474', 'AA2553', 'AA3548', 'AA3947', 'AA0546', 'AA3599', 'AA2917', 'AA3170', 'AA2850', 'AA4199', 'AA2890', 'AA2968', 'AA2903', 'AA3535', 'AA3910', 'AA3177', 'AA4642', 'AA3486', 'AA6578', 'AA5939', 'AA3708', 'AA7119', 'AA3598', 'AA3637', 'AA7685', 'AA2855', 'AA3127', 'AA2828', 'AA0102', 'AA3880', 'AA3596', 'AA3430', 'AA3674', 'AA6076', 'AA8978', 'AA3621', 'AA2856', 'AA3707', 'AA3070', 'AA3509', 'AA3476', 'AA3180', 'AA2829', 'AA2846', 'AA3441', 'AA2857', 'AA2208', 'AA3523', 'AA3633', 'AA3076', 'AA4372', 'AA3394', 'AA2977', 'AA2895', 'AA3559', 'AA3392', 'AA3053', 'AA3488', 'AA2938', 'AA9875', 'AA8459', 'AA3052', 'AA3766', 'AA3718', 'AA2896', 'AA3741', 'AA3588', 'AA2867', 'AA1160', 'AA3594', 'AA3813', 'AA3317', 'AA3000', 'AA0557', 'AA3022', 'AA3416', 'AA3604', 'AA3569', 'AA5617', 'AA3624', 'AA3605', 'AA2883', 'AA2951', 'AA3029', 'AA7030', 'AA2858', 'AA2953', 'AA2884', 'AA2952', 'AA5297', 'AA1353', 'AA0329', 'AA3336', 'AA1368', 'AA2778', 'AA3629', 'AA2970', 'AA3137', 'AA9132', 'AA8199', 'AA8041', 'AA3506', 'AA3577', 'AA7541', 'AA2765', 'AA3044', 'AA2901', 'AA3618', 'AA9022', 'AA1087', 'AA2986', 'AA4331', 'AA4259', 'AA5106', 'AA3675', 'AA2941', 'AA3685', 'AA9107', 'AA0063', 'AA2346', 'AA3673', 'AA3537', 'AA3176', 'AA6008', 'AA2956', 'AA2286', 'AA2879', 'AA3426', 'AA9791', 'AA4254', 'AA3582', 'AA3395', 'AA3085', 'AA3626', 'AA2974', 'AA3046', 'AA8077', 'AA9175', 'AA3305', 'AA2943', 'AA5153', 'AA3678', 'AA4255', 'AA2913', 'AA2904', 'AA3319', 'AA2852', 'AA3428', 'AA3496', 'AA2910', 'AA3542', 'AA3730', 'AA2293', 'AA1935', 'AA2899', 'AA2355', 'AA2428', 'AA3167', 'AA0269', 'AA3597', 'AA3358', 'AA2958', 'AA8267', 'AA3532', 'AA5736', 'AA3735', 'AA2872', 'AA3367', 'AA9094', 'AA2861', 'AA7368', 'AA3229', 'AA2948', 'AA3427', 'AA3590', 'AA3119', 'AA2979', 'AA8011']
tfidf = ['AA6736', 'AA3303', 'AA3432', 'AA3435', 'AA2905', 'AA3023', 'AA3186', 'AA3715', 'AA2831', 'AA2851', 'AA2945', 'AA3355', 'AA3630', 'AA1049', 'AA5939', 'AA3460', 'AA3596', 'AA2346', 'AA1300', 'AA3198', 'AA3206', 'AA3384', 'AA3398', 'AA6578', 'AA4331', 'AA1368', 'AA2977', 'AA3732', 'AA1902', 'AA3194', 'AA3210', 'AA3249', 'AA3208','AA2285', 'AA2286', 'AA2082', 'AA3121', 'AA7541', 'AA3195', 'AA3201', 'AA3216', 'AA3218', 'AA3416', 'AA3332', 'AA3348', 'AA3350', 'AA3351', 'AA3360', 'AA2869', 'AA2933', 'AA3189', 'AA3284', 'AA3409', 'AA1978', 'AA3197', 'AA3202', 'AA3209', 'AA3320', 'AA3387', 'AA3434','AA3443', 'AA3419', 'AA3425', 'AA3385', 'AA3407', 'AA3415', 'AA1218','AA6342']
one= ['AA2765', 'AA2778', 'AA2825', 'AA2826', 'AA2784', 'AA2834', 'AA4209', 'AA2843', 'AA8011', 'AA5706', 'AA2844', 'AA2771', 'AA2758', 'AA2847', 'AA2854', 'AA2856', 'AA2831', 'AA2553', 'AA2866', 'AA2867', 'AA2871', 'AA6578', 'AA2872', 'AA2851', 'AA2846', 'AA2879', 'AA2881', 'AA2883', 'AA2886', 'AA2888', 'AA4291', 'AA2892', 'AA2893', 'AA2885', 'AA2868', 'AA2900', 'AA2901', 'AA2902', 'AA6370', 'AA2905', 'AA2906', 'AA2913', 'AA2912', 'AA2917', 'AA2920', 'AA1377', 'AA4878', 'AA2938', 'AA2082', 'AA2933', 'AA0269', 'AA2368', 'AA2898', 'AA2958', 'AA2965', 'AA2969', 'AA4438', 'AA2974', 'AA2983', 'AA2985', 'AA2987', 'AA1902', 'AA2989', 'AA2990', 'AA2993', 'AA3000', 'AA3001', 'AA3002', 'AA3004', 'AA3009', 'AA3013', 'AA3014', 'AA3015', 'AA1368', 'AA2945', 'AA3020', 'AA3021', 'AA3022', 'AA3023', 'AA3024', 'AA2159', 'AA5609', 'AA3029', 'AA3037', 'AA3038', 'AA2992', 'AA3041', 'AA3045', 'AA3050', 'AA3053', 'AA3055', 'AA3056', 'AA7523', 'AA3947', 'AA3062', 'AA5939', 'AA8683', 'AA2643', 'AA3075', 'AA3076', 'AA3077', 'AA3085', 'AA9791', 'AA0102', 'AA3507', 'AA1300', 'AA3086', 'AA3064', 'AA7944', 'AA3139', 'AA3127', 'AA1087', 'AA4255', 'AA3175', 'AA3172', 'AA0063', 'AA6736', 'AA3185', 'AA3186', 'AA3188', 'AA3192', 'AA3195', 'AA3196', 'AA3197', 'AA3198', 'AA3199', 'AA3200', 'AA3201', 'AA3202', 'AA2908', 'AA3204', 'AA3205', 'AA3209', 'AA3212', 'AA3213', 'AA3215', 'AA3216', 'AA3217', 'AA3218', 'AA3219', 'AA3220', 'AA3221', 'AA3224', 'AA3225', 'AA3226', 'AA3227', 'AA3228', 'AA3231', 'AA3232', 'AA3235', 'AA3236', 'AA3237', 'AA3238', 'AA3241', 'AA3242', 'AA3243', 'AA3246', 'AA3248', 'AA3250', 'AA3260', 'AA3262', 'AA1972', 'AA3265', 'AA3267', 'AA3269', 'AA3270', 'AA3271', 'AA3272', 'AA3278', 'AA3283', 'AA3189', 'AA3285', 'AA3286', 'AA2531', 'AA3291', 'AA3300', 'AA3302', 'AA3304', 'AA3311', 'AA3314', 'AA0947', 'AA3309', 'AA3315', 'AA0743', 'AA3320', 'AA3328', 'AA3331', 'AA3332', 'AA3348', 'AA3350', 'AA3351', 'AA3352', 'AA3357', 'AA3360', 'AA3369', 'AA3374', 'AA3381', 'AA3387', 'AA3390', 'AA3394', 'AA3395', 'AA3400', 'AA8267', 'AA5585', 'AA9790', 'AA3408', 'AA3412', 'AA3413', 'AA3416', 'AA3419', 'AA3422', 'AA3423', 'AA3427', 'AA3429', 'AA3432', 'AA3433', 'AA3434', 'AA3435', 'AA3438', 'AA3441', 'AA3443', 'AA3444', 'AA3447', 'AA3451', 'AA3452', 'AA3453', 'AA3454', 'AA3456', 'AA3457', 'AA3458', 'AA3460', 'AA6007', 'AA3476', 'AA3284', 'AA3409', 'AA3319', 'AA3339', 'AA3487', 'AA1978', 'AA4905', 'AA3355', 'AA3368', 'AA3373', 'AA3375', 'AA3379', 'AA3382', 'AA1134', 'AA3206', 'AA5408', 'AA5192', 'AA3496', 'AA3384', 'AA3385', 'AA3389', 'AA3398', 'AA3208', 'AA8041', 'AA9875', 'AA3407', 'AA3414', 'AA3415', 'AA1218', 'AA3439', 'AA3506', 'AA3522', 'AA7119', 'AA1410', 'AA3486', 'AA3538', 'AA7685', 'AA3523', 'AA4642', 'AA3559', 'AA8857', 'AA3563', 'AA4763', 'AA3566', 'AA3580', 'AA6342', 'AA3121', 'AA3582', 'AA3590', 'AA0422', 'AA3593', 'AA3587', 'AA2769', 'AA7601', 'AA3597', 'AA3600', 'AA3604', 'AA3591', 'AA3605', 'AA6008', 'AA3610', 'AA3584', 'AA3618', 'AA3624', 'AA3628', 'AA3631', 'AA3632', 'AA1707', 'AA3634', 'AA3636', 'AA9802', 'AA3630', 'AA2699', 'AA3635', 'AA3673', 'AA0305', 'AA3675', 'AA1661', 'AA3880', 'AA3558', 'AA0689', 'AA8395', 'AA1979', 'AA9236', 'AA3685', 'AA3621', 'AA3689', 'AA3691', 'AA2346', 'AA3701', 'AA4200', 'AA3703', 'AA8549', 'AA3690', 'AA3707', 'AA3708', 'AA3705', 'AA9325', 'AA3688', 'AA3715', 'AA2977', 'AA3723', 'AA7518', 'AA0340', 'AA3741', 'AA8130', 'AA9951', 'AA3768', 'AA3767', 'AA7368', 'AA0552', 'AA3714', 'AA5645', 'AA3801', 'AA9929', 'AA3766', 'AA3813']
clusterSet = set(cluster)
one = set(one)
clusterSet = clusterSet - one
cluster = list(clusterSet)



with open("final.data", "r") as data:
    lines = data.readlines()
    for line in lines:
        line = line.split()
        if line[1] == "未開戶":
            noHu.append(line[0])
        elif line[1] == "有交易":
            trade.append(line[0])
        elif line[1] == "無交易":
            noTrade.append(line[0])

clusternoHuCount = 0
clusternoTradeCount = 0
clustertradeCount = 0
for id in cluster:
    if id in noHu:
        clusternoHuCount += 1
    elif id in noTrade:
        clusternoTradeCount += 1
    elif id in trade:
        clustertradeCount += 1

print("Cluster no Hu: ", clusternoHuCount/len(cluster))
print("Cluster Trade rate: ", clustertradeCount/len(cluster))
print("Cluster Has account rate: ", (clusternoTradeCount + clustertradeCount)/len(cluster))

keywordnoHuCount = 0
keywordnoTradeCount = 0
keywordtradeCount = 0
for id in keyword:
    if id in noHu:
        keywordnoHuCount += 1
    elif id in noTrade:
        keywordnoTradeCount += 1
    elif id in trade:
        keywordtradeCount += 1

print("Keyword no Hu: ", keywordnoHuCount/len(keyword))
print("Keyword Trade rate: ", keywordtradeCount/len(keyword))
print("Keyword Has account rate: ", (keywordnoTradeCount + keywordtradeCount)/len(keyword))


tfidfnoHuCount = 0
tfidfnoTradeCount = 0
tfidftradeCount = 0
for id in tfidf:
    if id in noHu:
        tfidfnoHuCount += 1
    elif id in noTrade:
        tfidfnoTradeCount += 1
    elif id in trade:
        tfidftradeCount += 1

print("TF-IDF no Hu: ", tfidfnoHuCount/len(tfidf))
print("TF-IDF Trade rate: ", tfidftradeCount/len(tfidf))
print("TF-IDF Has account rate: ", (tfidfnoTradeCount + tfidftradeCount)/len(tfidf))


