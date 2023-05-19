##### Load all the modules from utils
import time
import joblib
from utils import *
import land_cover_clustering as lcc
import models



##### Cluster train/test data and assign clusters of pixels to land cover types.
if __name__ == '__main__':
    start = time.time()
    ### Train set clustering
    DATA_DIR = "../data-in"
    IN_DIR = "train"
    train_data, bands = lcc.lclu_clustering(DATA_DIR, IN_DIR)
    #train_data.to_csv("../data-out/train-lcc-sentibel2.csv", index=False)

    ### Test set clustering
    DATA_DIR = "../data-in"
    IN_DIR = "test"
    test_data, _ = lcc.lclu_clustering(DATA_DIR, IN_DIR)
    #test_data.to_csv("../data-out/test-lcc-sentibel2.csv", index=False)


    ########## Assessing the Land Cover Clustering model with LGBM

    ##### Data extraction for LGBM training and testing
    ### Training set
    train_x = train_data.loc[:, bands].copy()
    train_y = train_data.loc[:, "classes"].copy()

    ### Testing set
    test_x = test_data.loc[:, bands].copy()
    test_y = test_data.loc[:, "classes"].copy()

    ##### Training of the LGBM Classifier
    lgbm_clf = models.lgbm_clissifier(train_x, train_y,
                                    mb_round=150, verbose_eval=10)

    ##### Validation

    print(f"\n\n\nValidation of the trained LGBM classifier ...\n\n")
    _class_probs = lgbm_clf.predict(test_x) # Predict probs of classes
    pred_labels = np.argmax(_class_probs, axis=1) # Get pred_labels




    ########## Get the performance

    ##### Land cover types or classes:
    lc_types = {0: "Unlabel",
                1: "Water",
                2: "Nat. Vegetation",
                3: "Agr. Fields",
                4: "Bare Ground",
                5: "Urban"}

    lc_classes = pd.DataFrame(lc_types, index=[1]).T.iloc[:, 0].to_list()

    ##### Accuracy and classification report
    jacc_score = round(jaccard_score(test_y, pred_labels)*100, 2)
    jscore_info = colored(jacc_score, "green")
    print(f"{' '*2} Mean of IoU or Jaccard similarity score: {jscore_info}\n")
    jscore_info = f"\nJaccard similarity score or mean of IoU: {jacc_score}%\n\n\n"


    print(f"{' '*2} Prepare the clssification report and save\n")
    clr_title = "\nLGBM Classification Report:\n\n"
    class_report = classification_report(test_y, pred_labels,
                                        digits=4, target_names=lc_classes)
    print(class_report)

    with open("../data-out/classification_report.txt", "w") as clr:
        clr.writelines(jscore_info)
        clr.writelines(clr_title)
        clr.writelines(class_report)

    ##### Save the trained model
    print(f"\n\n{' '*2} Saving the trained LGBM model for further use\n")
    sname = "../data-out/models/lgbm-lcc-sentinel2.joblib"
    joblib.dump(lgbm_clf, sname)


    ##### Get the confusion matrix
    print(f"\n{' '*2} Confusion Matrix & Feature Importance\n")
    conf_matrix = confusion_matrix(test_y, pred_labels)

    ##### Feature importance and Heat Map
    sfg_info = f"\n{' '*2} Please Close the Figure on the Screen to Continue \n"
    print(colored(sfg_info, "green"))

    fig = plot_cfmatrix_fimp_lgbm(lgbm_clf, test_y,
                                pred_labels, lc_classes, bands)


    ### Save the figure
    sfig = "feat_imp-conf_matrix_lcc_lgbm.png"
    fig.savefig(f"../figures/{sfig}", dpi=300, bbox_inches="tight")


    print(f"\n{' '*2} Saving the Feature Importance\n")
    fimp = pd.DataFrame(sorted(zip(lgbm_clf.feature_importance(), bands)),
                        columns=["Value", "Feature"])
    fimp = fimp.set_index("Feature")/lgbm_clf.feature_importance().sum()
    fimp.reset_index(inplace=True)
    fimp.to_csv("../data-out/feature_importance_lgbm_lcc.csv", index=False)


    end = time.time()
    time_col = round((end - start)/60, 2)

    out_str = f"\n\nThe whole time this LC-Clustering-Classification is: {time_col} minutes\n\n"
    print(out_str)
    print("ALL DONE!\n")






