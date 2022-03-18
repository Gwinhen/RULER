import os
import sys
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix

sys.path.append(os.getcwd())

"""To evaluate an approach with the metrics of aod, spd, DP, DPR, EO"""


class Metrics:
    def __init__(self, x_true, y_true, y_pred, favorable_class, privileged_group, protected_attribs):
        """
        :param y_true: npy, the original class
        :param y_pred: npy, the predict class
        :param favorable_class: int, the favorable_class is a value of our label
        :param privileged_group:int, a value of our protected_attribs
        :param protected_attribs:int, the index of protected_attribs
        """
        # print('aod cwd {}\n aod path {}'.format(os.getcwd(), sys.path))
        self.x_true = x_true
        self.y_true = y_true
        self.y_pred = y_pred
        self.favorable_class = favorable_class
        self.privileged_group = privileged_group
        self.protected_attribs = protected_attribs
        spd, dp, dpr = self._compute_spd_dp_dpr()
        self.results = {'spd': spd,
                        'aod': self._compute_aod(),
                        'eo': self._compute_eo(),
                        'dp': dp,
                        'dpr': dpr}

    def _compute_spd_dp_dpr(self):
        pred_data = np.hstack([self.x_true, self.y_pred])
        pred_data[:, self.protected_attribs] = (pred_data[:, self.protected_attribs] == self.privileged_group)
        pred_data[:, self.protected_attribs] = pred_data[:, self.protected_attribs].astype(np.int32)
        favo_privi = pred_data[
            np.logical_and(
                pred_data[:, -1] == self.favorable_class,
                pred_data[:, self.protected_attribs] == 1
            )
        ]
        favo_unprivi = pred_data[
            np.logical_and(
                pred_data[:, -1] == self.favorable_class,
                pred_data[:, self.protected_attribs] == 0
            )
        ]

        if len(pred_data[pred_data[:, self.protected_attribs] == 0]) == 0:
            Pr_unprivi = 0
        else:
            Pr_unprivi = len(favo_unprivi) / len(pred_data[pred_data[:, self.protected_attribs] == 0])
        if len(pred_data[pred_data[:, self.protected_attribs] == 1]) == 0:
            Pr_privi = 0
        else:
            Pr_privi = len(favo_privi) / len(pred_data[pred_data[:, self.protected_attribs] == 1])
        # print(Pr_unprivi, Pr_privi)
        spd = Pr_unprivi - Pr_privi
        dp = abs(spd)
        # Pr(yhat=1|S=1) / Pr(yhat=1|S=0)
        dpr = 0 if Pr_unprivi == 0 else Pr_privi / Pr_unprivi
        print('Pr_privi is {}.Pr_unprivi is {}'.format(Pr_privi, Pr_unprivi))
        return spd, dp, dpr

    def _compute_aod(self):
        pred_true_data = np.hstack([self.x_true, self.y_true, self.y_pred])
        pred_true_data[:, self.protected_attribs] = (pred_true_data[:, self.protected_attribs] == self.privileged_group)
        pred_true_data[:, self.protected_attribs] = pred_true_data[:, self.protected_attribs].astype(np.int32)
        # formula aod = 0.5*[(FPR_unprivi-FPR_privi)+(TPR_unprivi-TPR_privi)]
       
        pred_true_data_privi = pred_true_data[
            pred_true_data[:, self.protected_attribs] == 1
            ]
        pred_true_data_unprivi = pred_true_data[
            pred_true_data[:, self.protected_attribs] == 0
            ]
        y_true_unprivi = pred_true_data_unprivi[:, -2]
        y_pred_unprivi = pred_true_data_unprivi[:, -1]
        Confusion_unprivi = np.array(confusion_matrix(y_true=y_true_unprivi, y_pred=y_pred_unprivi, labels=[0, 1]))

        favo = self.favorable_class
        unfavo = (self.favorable_class + 1) % 2
        TPR_unprivi = 0 if sum(Confusion_unprivi[:, favo]) == 0 \
            else Confusion_unprivi[favo, favo] / sum(Confusion_unprivi[:, favo])
        FPR_unprivi = 0 if sum(Confusion_unprivi[:, unfavo]) == 0 \
            else Confusion_unprivi[favo, unfavo] / sum(Confusion_unprivi[:, unfavo])

        
        y_true_privi = pred_true_data_privi[:, -2]
        y_pred_privi = pred_true_data_privi[:, -1]
        Confusion_privi = np.array(confusion_matrix(y_true=y_true_privi, y_pred=y_pred_privi, labels=[0, 1]))
        TPR_privi = 0 if sum(Confusion_privi[:, 1]) == 0 else Confusion_privi[1, 1] / sum(Confusion_privi[:, 1])
        FPR_privi = 0 if sum(Confusion_privi[:, 0]) == 0 else Confusion_privi[1, 0] / sum(Confusion_privi[:, 0])

        aod = 0.5 * ((FPR_unprivi - FPR_privi) + (TPR_unprivi - TPR_privi))
        return aod

    def _compute_eo(self):
        pred_true_data = np.hstack([self.x_true, self.y_true, self.y_pred])
       
        pred_true_data[:, self.protected_attribs] = (pred_true_data[:, self.protected_attribs] == self.privileged_group)
        pred_true_data[:, self.protected_attribs] = pred_true_data[:, self.protected_attribs].astype(np.int32)
        # Formula EO = |P(Y_hat=1|S=0,Y=1) - P(Y_hat=1|S=1, Y=1)|
       
        S0Y1 = pred_true_data[
            np.logical_and(
                pred_true_data[:, self.protected_attribs] == 0,
                pred_true_data[:, -2] == self.favorable_class
            )
        ]
        S1Y1 = pred_true_data[
            np.logical_and(
                pred_true_data[:, self.protected_attribs] == 1,
                pred_true_data[:, -2] == self.favorable_class
            )
        ]
       
        left = 0 if len(S0Y1) == 0 else len(S0Y1[S0Y1[:, -1] == self.favorable_class]) / len(S0Y1)
      
        right = 0 if len(S1Y1) == 0 else len(S1Y1[S1Y1[:, -1] == self.favorable_class]) / len(S1Y1)
     
        eo = abs(left - right)
        return eo

    def _compute_dp(self):
        
        dp = abs(self._compute_spd())
        return dp

    def _compute_dpr(self):
        
        pred_data = np.hstack([self.x_true, self.y_pred])
       
        pred_data[:, self.protected_attribs] = (pred_data[:, self.protected_attribs] == self.privileged_group)
        pred_data[:, self.protected_attribs] = pred_data[:, self.protected_attribs].astype(np.int32)

        
        favo_privi = pred_data[
            np.logical_and(
                pred_data[:, -1] == self.favorable_class,
                pred_data[:, self.protected_attribs] == 1
            )
        ]
       
        favo_unprivi = pred_data[
            np.logical_and(
                pred_data[:, -1] == self.favorable_class,
                pred_data[:, self.protected_attribs] == 0
            )
        ]

      
        Pr_unprivi = len(favo_unprivi) / len(pred_data[pred_data[:, self.protected_attribs] == 0])
        Pr_privi = len(favo_privi) / len(pred_data[pred_data[:, self.protected_attribs] == 1])
        dpr = Pr_privi / Pr_unprivi
        return dpr

    def get_metrics(self):
        return self.results


def test():
    
    x_true = np.array([
        [0, 1, 2, 3, 4],
        [1, 1, 2, 3, 4],
        [2, 1, 2, 3, 4],
    ])
    y_true = np.array([0, 1, 0])
    y_pred = np.array([1, 0, 0])
    metrics = Metrics(x_true=x_true,
                      y_true=y_true.reshape(-1, 1),
                      y_pred=y_pred.reshape(-1, 1),
                      favorable_class=1, privileged_group=2, protected_attribs=0)
    print(metrics.get_metrics())
   
def test_h5_model(dataset):
    """
    COMPAS.race: {2(race) :1(Caucasian)} -> {0: low score}
    Adult.race: {6(race): 0(white)} -> {1 : income>50k}
    Adult.gender: {7(gender): 1(male)} -> {1 : income > 50k}
    """
    x_true = np.load('../data/PGD_dataset/{}/x_test.npy'.format(dataset))
    y_true = np.load('../data/PGD_dataset/{}/y_test.npy'.format(dataset))
    ori_model = keras.models.load_model('./evaulate_specific_model_h5/best_models/{}_model.h5'.format(dataset))
    ADF_model = keras.models.load_model(
        './evaulate_specific_model_h5/best_models/{}_ADF_retrained_model.h5'.format(dataset))
    EIDIG_model = keras.models.load_model(
        './evaulate_specific_model_h5/best_models/{}_EIDIG_5_retrained_model.h5'.format(dataset))
    favorable_class = 1
    privileged_group = 1
    protected_attribs = 7
    # print(pred.shape, x_true.shape, y_true.shape)

    # original model
    pred = np.array(ori_model.predict(x_true) > 0.5, dtype=int)
    metrics = Metrics(
        x_true=x_true,
        y_true=y_true.reshape(-1, 1),
        y_pred=pred.reshape(-1, 1),
        favorable_class=favorable_class,
        privileged_group=privileged_group,
        protected_attribs=protected_attribs
    ).get_metrics()
    print('ori_model is \n{}'.format(metrics))

    # ADF model
    pred = np.array(ADF_model.predict(x_true) > 0.5, dtype=int)
    # print(pred.shape, x_true.shape, y_true.shape)
    metrics = Metrics(
        x_true=x_true,
        y_true=y_true.reshape(-1, 1),
        y_pred=pred.reshape(-1, 1),
        favorable_class=favorable_class,
        privileged_group=privileged_group,
        protected_attribs=protected_attribs
    ).get_metrics()
    print('ADF_model is \n{}'.format(metrics))

    # EIDIG model
    pred = np.array(EIDIG_model.predict(x_true) > 0.5, dtype=int)
    # print(pred.shape, x_true.shape, y_true.shape)
    metrics = Metrics(
        x_true=x_true,
        y_true=y_true.reshape(-1, 1),
        y_pred=pred.reshape(-1, 1),
        favorable_class=favorable_class,
        privileged_group=privileged_group,
        protected_attribs=protected_attribs
    ).get_metrics()
    print('EIDIG_model is \n{}'.format(metrics))


if __name__ == "__main__":
    test_h5_model('adult')
