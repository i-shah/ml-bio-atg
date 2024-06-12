import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from .ml_graph_classifier_oop import GraphModel

class GraphModelPlotter(GraphModel):

    def __init__(self,*args,**kwargs):
        super(GraphModelPlotter,self).__init__(*args,**kwargs)
        
    def _get_assay_data(self,assay,preds=None):
        df = pd.DataFrame()
        assay_n = self.assay_list.index(assay)


        if preds:
            dataset = self.test_dataset
            z = self.test_z

        else:
            dataset = self.train_dataset
            z = self.train_z

        df['feature'] = dataset.X
        df['labels'] = dataset.y[:,assay_n]
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]
        return df
        


    def plot_kde(self,assay):
        """
        Parameters
        ----------

        assay: str

        """

        df = self._get_assay_data(assay)
        g = sns.kdeplot(data = df,x='comp-1',y='comp-2',hue ='labels',fill=True,levels=20,thresh=0.15,alpha=.5,palette=['b','r'])
        #g = sns.scatterplot(data = df,x='comp-1',y='comp-2',hue ='labels',palette=['b','r'])
        g.set_title(assay,fontsize=20)

        return df


    def plot_scatter(self,assay,labeled=True):
        """
        Parameters
        ----------
        assay: str
        """
        assay_n = self.assay_list.index(assay)

        if labeled:
            df = self._get_assay_data(assay,True)
            preds = self.test_predictions[:,assay_n][:,1]
        else:
            df = self.get_unlabeled_assay_data(assay)
            preds = df[f'probability active {assay}']
            
        g = sns.scatterplot(data = df,x='comp-1',y='comp-2',c=preds,cmap='coolwarm',vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=None)
        sm.set_array([])
        g.figure.colorbar(sm)
        g.get_legend().remove()
        

    def plot_train_test(self,assay):
        self.plot_kde(assay)
        self.plot_scatter(assay)


    def plot_train_unlabeled(self,assay):
        self.plot_kde(assay)
        self.plot_scatter(assay,False)