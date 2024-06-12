from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from IPython.display import display
import matplotlib
import matplotlib.cm as cm
import dgl
import torch
import matplotlib.pyplot as plt
import numpy as np
import deepchem as dc
from ..db.graph_utils import *
from .ml_graph_classifier import *
import seaborn as sns


def draw_mol_weights(DTXSID,model,text,timestep):
    """
    Parameters
    ----------

    DTXSID: str

    model: dc.model

    timestep: int 

    """
    smiles = get_smiles(DTXSID)
    chm_name = get_chm_name(DTXSID)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    graph = featurizer.featurize(smiles)[0]
    g = graph.to_dgl_graph()
    bg = dgl.batch([g])
    _ , _ , _, atom_weights = model.model(bg)
    atom_weights = atom_weights[timestep]
    min_value = torch.min(atom_weights)
    max_value = torch.max(atom_weights)
    atom_weights = (atom_weights - min_value) / (max_value - min_value)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = cm.get_cmap('bwr')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i].data.item()) for i in range(bg.number_of_nodes())}
    
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(10)
    op = drawer.drawOptions()
    
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(bg.number_of_nodes()),
                             highlightBonds=[],
                             highlightAtomColors=atom_colors,
                             legend=DTXSID + '\t' +  chm_name[:18] + '\n' + text)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    if torch.cuda.is_available():
        atom_weights = atom_weights.to('cpu')
        
    a = np.array([[0,1]])
    plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap="bwr")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.2])
    plt.colorbar(orientation='horizontal', cax=cax)
    plt.show()
    return (Chem.MolFromSmiles(smiles), atom_weights.data.numpy(), svg)
    
def plot_space(assay,assay_list,data,z,y_pred=None):
    """
    Parameters
    ----------

    assay: str

    assay_list: list[str]

    train_z: np.ndarray

    z: np.ndarray

    y_pred: np.ndarray

    """
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize':(7,5)})
    assay_n = assay_list.index(assay)
    df = pd.DataFrame()
    df['feature'] = data.X
    df['labels'] = data.y[:,assay_n]
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    
    if y_pred is not None:
        preds = y_pred[:,assay_n][:,1]
        norm = plt.Normalize(preds.min(), preds.max())
        g = sns.scatterplot(data = df,x='comp-1',y='comp-2',c=preds,cmap='coolwarm',vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=None)
        sm.set_array([])
        g.figure.colorbar(sm)
        g.get_legend().remove()
    else:
        g = sns.kdeplot(data = df,x='comp-1',y='comp-2',hue ='labels',fill=True,levels=20,thresh=.1,palette=['b','r'])


    g.set_title(assay,fontsize=20)

    return df

def plot_labeled(ASSAY_NAME,assay_list,train_dataset,test_dataset,train_z,test_z,y_pred):
    plot_space(ASSAY_NAME,assay_list,train_dataset,train_z)
    plot_space(ASSAY_NAME,assay_list,test_dataset,test_z,y_pred)


def plot_unlabeled(ASSAY_NAME,data):
    preds = predict_unlabeled(ASSAY_NAME, assay_list, data, model, train_z)
    plot_space(ASSAY_NAME, assay_list, train_dataset, train_z)
