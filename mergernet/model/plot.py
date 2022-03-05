import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay



plot_params = {
  # 'text.usetex': True,
  # 'text.latex.preamble': r'\usepackage{times}',
  # 'font.family': 'serif',
  # 'font.serif': 'Times',
  'font.family': 'DejaVu Sans',
  'legend.fontsize': 13,
  'axes.labelsize': 15,
  # 'axes.grid': True,
  'axes.grid.axis': 'both',
  'grid.linestyle': '-',
  'grid.alpha': 0.35,
  'grid.linewidth': 0.8,
  'xtick.labelsize': 13,
  'ytick.labelsize': 13,
  'savefig.bbox': 'tight',
  'savefig.pad_inches': 0.01
}

mpl.rcParams.update(plot_params)



class Serie:
  def __init__(self, serie=None, label=None, **kwargs):
    self.label = label
    self.kwargs = kwargs
    self.serie = np.array(serie)

  def has_std(self):
    return len(self.serie.shape) > 1

  def get_std(self):
    if self.has_std():
      return np.std(self.serie)
    else:
      return np.zeros(self.series.shape)

  def get_serie(self, median=False):
    if len(self.serie.shape) == 1:
      return self.serie
    else:
      if median:
        return np.median(self.serie)
      else:
        return np.mean(self.serie)



def train_metrics(
  mean_series,
  error_series,
  xlim=None,
  ylim=None,
  legend_loc='best',
  filename=None
):
  plt.figure(1, figsize=(8,8))

  for mean, error in zip(mean_series, error_series):
    x = np.arange(1, len(mean.serie) + 1)
    plt.plot(x, mean.serie, label=mean.label, **mean.kwargs)
    plt.fill_between(x, mean.serie - error.serie, mean.serie + error.serie, **error.kwargs)

  plt.xlabel('Épocas')
  plt.ylabel('Custo')
  plt.legend(loc=legend_loc)

  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)
  if filename is not None:
    plt.savefig(filename)
  plt.show()



def roc(
  true_series,
  pred_series,
  curves,
  zoom_range=None,
  zoom_pos=None,
  filename=None,
  show=True
):
  """Plot ROC curve of a model
  Parameters
  ----------
  true_series: list of Serie
    The series of true labels.
    Shape of each serie: (n, n_class),
    where rep is the number of repetitions (std curve computation),
    n is the number of examples and n_class is the number of classes.
  pred_series: list of Serie
    The series of predictions.
    Shape of each serie: (n, n_class),
    where rep is the number of repetition (std curve computation),
    n is the number of examples and n_class is the number of classes.
  curves: list
    Values can include: 0 ... n_class, "macro" or "micro".
    If curve is a int, then a curve of the specific class will be ploted.
    If curve is "macro", the macro-average of all classes will be computed.
    If curve is "micro", the micro-average of all classes will be computed.

  Notes
  -----
  assert len(true_series) == len(pred_series) == len(curves)
  """
  _, ax = plt.subplots(figsize=(8,8))
  fpr_series, tpr_series, roc_auc_series = [], [], []

  for j, curve in enumerate(curves):
    fpr, tpr, roc_auc = {}, {}, {}
    y_true = true_series[j].serie
    y_pred = pred_series[j].serie
    n_classes = y_true.shape[-1]

    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    if curve == 'micro':
      fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_pred.ravel())
      roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    if curve == 'macro':
      # Aggregate all false positive rates
      all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

      # Interpolate all ROC curves at this points
      mean_tpr = np.zeros_like(all_fpr)
      for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

      # Average it and compute AUC
      mean_tpr /= n_classes
      fpr['macro'] = all_fpr
      tpr['macro'] = mean_tpr
      roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    fpr_series.append(fpr[curve])
    tpr_series.append(tpr[curve])
    roc_auc_series.append(roc_auc[curve])

  # Plot all ROC curves
  for i in range(len(curves)):
    label = true_series[i].label or pred_series[i].label or ''
    label += f' (AUC = {roc_auc_series[i]:0.3f})'
    ax.plot(
      fpr_series[i],
      tpr_series[i],
      label=label,
      linewidth=1.45,
      **true_series[i].kwargs
    )

  if zoom_range is not None:
    axins = ax.inset_axes(zoom_pos)
    for i in range(len(curves)):
      axins.plot(
        fpr_series[i],
        tpr_series[i],
        linewidth=1.45,
        **true_series[i].kwargs
      )
    axins.set_xticks((max(0, zoom_range[0]), min(1, zoom_range[1])))
    axins.set_yticks((max(0, zoom_range[2]), min(1, zoom_range[3])))
    axins.set_xlim(zoom_range[0], zoom_range[1])
    axins.set_ylim(zoom_range[2], zoom_range[3])
    axins.tick_params(axis='both', which='major', labelsize=11)
    ax.indicate_inset_zoom(axins, label=None)

  ax.plot([0, 1], [0, 1], '--', c='k')
  ax.set_ylim(0, 1.01)
  ax.set_xlim(-0.01, 1)
  ax.set_xlabel('FPR')
  ax.set_ylabel('TPR')
  ax.legend()

  if filename is not None:
    plt.savefig(filename)
  if show:
    plt.show()



def data_distribution(
  train_series,
  test_series,
  blind_series,
  min_range=None,
  max_range=None,
  xlabel=None,
  ylabel=None,
  loc='best',
  vline=None,
  filename=None
):
  min_range = min_range or min([s.min() for s in [train_series, test_series, blind_series]])
  max_range = max_range or max([s.max() for s in [train_series, test_series, blind_series]])

  common_args = {
    'bins': 30,
    'alpha': 0.9,
    'histtype': 'step',
    'linewidth': 1.5,
    'range': (min_range, max_range)
  }

  plt.figure(figsize=(8, 5))
  plt.hist(
    train_series,
    weights=np.ones(len(train_series))/len(train_series),
    label='Treinamento e Validação',
    ec='tab:red',
    **common_args
  )
  plt.hist(
    test_series,
    weights=np.ones(len(test_series))/len(test_series),
    label='Teste',
    ec='tab:blue',
    **common_args
  )
  plt.hist(
    blind_series,
    weights=np.ones(len(blind_series))/len(blind_series),
    label='Blind',
    ec='tab:green',
    **common_args
  )

  if vline is not None:
    ylim = plt.ylim()
    plt.vlines(vline, 0, 1, ec='black', alpha=0.5, ls='-', lw=0.8)
    plt.ylim(ylim)

  plt.gca().yaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, _: f'${int(x * 100)}\%$')
  )

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(loc=loc)

  if filename:
    plt.savefig(filename)
  plt.show()



def color_color(table_e, table_s, xlim=None, ylim=None, relative=False, filename=None):
  # table_e = table_e[table_e.specsfr_tot_p50 > -999]
  # table_s = table_s[table_s.specsfr_tot_p50 > -999]

  x_e = table_e.uJAVA_aper - table_e.g_aper
  # y_e = table_e.specsfr_tot_p50
  y_e = table_e.g_aper - table_e.r_aper
  # y_e = table_e.r_auto - table_e.z_auto
  x_s = table_s.uJAVA_aper - table_s.g_aper
  y_s = table_s.g_aper - table_s.r_aper
  # y_s = table_s.specsfr_tot_p50
  # y_s = table_s.r_auto - table_s.z_auto

  left, width = 0.1, 0.65
  bottom, height = 0.1, 0.65
  spacing = 0.005

  rect_scatter = [left, bottom, width, height]
  rect_histx = [left, bottom + height + spacing, width, 0.2]
  rect_histy = [left + width + spacing, bottom, 0.2, height]

  plt.figure(figsize=(8, 8))

  ax_scatter = plt.axes(rect_scatter)
  ax_scatter.tick_params(direction='in', top=True, right=True, labelsize=13)
  ax_histx = plt.axes(rect_histx)
  ax_histx.tick_params(direction='in', labelbottom=False, labelsize=12.5)
  ax_histy = plt.axes(rect_histy)
  ax_histy.tick_params(direction='in', labelleft=False, labelsize=12.5)

  ax_scatter.scatter(x_s, y_s, color='tab:blue', linewidths=0, label='Espiral', alpha=0.66, s=20)
  ax_scatter.scatter(x_e, y_e, color='tab:red', linewidths=0, label='Elíptica', alpha=0.66, s=20)

  if xlim is not None:
    ax_scatter.set_xlim(xlim)
  if ylim is not None:
    ax_scatter.set_ylim(ylim)

  bins_x = np.linspace(ax_scatter.get_xlim()[0], ax_scatter.get_xlim()[1], 40)
  bins_y = np.linspace(ax_scatter.get_ylim()[0], ax_scatter.get_ylim()[1], 40)
  if relative:
    ax_histx.hist(x_e, bins=bins_x, weights=np.ones(len(x_e))/len(x_e), alpha=0.55, fc='tab:red', ec='white')
    ax_histx.hist(x_s, bins=bins_x, weights=np.ones(len(x_s))/len(x_s), alpha=0.55, fc='tab:blue', ec='white')
    ax_histy.hist(y_e, bins=bins_y, weights=np.ones(len(y_e))/len(y_e), alpha=0.55, fc='tab:red', ec='white', orientation='horizontal')
    ax_histy.hist(y_s, bins=bins_y, weights=np.ones(len(y_s))/len(y_s), alpha=0.55, fc='tab:blue', ec='white', orientation='horizontal')
    ax_histx.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f'${int(x * 100)}\%$'))
    ax_histy.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f'${int(x * 100)}\%$'))
  else:
    ax_histx.hist(x_e, bins=bins_x, alpha=0.55, fc='tab:red', ec='white')
    ax_histx.hist(x_s, bins=bins_x, alpha=0.55, fc='tab:blue', ec='white')
    ax_histy.hist(y_e, bins=bins_y, alpha=0.55, fc='tab:red', ec='white', orientation='horizontal')
    ax_histy.hist(y_s, bins=bins_y, alpha=0.55, fc='tab:blue', ec='white', orientation='horizontal')

  ax_histx.set_xlim(ax_scatter.get_xlim())
  ax_histy.set_ylim(ax_scatter.get_ylim())

  ax_scatter.set_axisbelow(True)
  ax_histx.set_axisbelow(True)
  ax_histy.set_axisbelow(True)

  ax_scatter.set_xlabel('$u_{JAVA} - g$')
  ax_scatter.set_ylabel('$g - r$')
  ax_scatter.legend(framealpha=0.7, handletextpad=0.1)

  plt.setp(ax_histx.get_yticklabels()[0], visible=False)
  plt.setp(ax_histy.get_xticklabels()[0], visible=False)

  # ax_scatter.invert_yaxis()
  # ax_histy.invert_yaxis()

  if filename is not None:
    plt.savefig(filename, facecolor='white')

  plt.show()



def conf_matrix(y_true, y_pred, one_hot: bool = False, labels: Sequence[str] = None):
  if one_hot:
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

  cm = confusion_matrix(y_true, y_pred)
  cm_display = ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap='Blues_r')
  return cm_display.ax_



def mag_class_distribution(
  df: pd.DataFrame,
  n_folds: int,
  n_bins: int,
  color_map: dict,
  label_map: dict = None,
  xlabel: str = 'r mag',
  ylabel: str = 'count',
  title: str = None,
  figsize: Tuple[float, float] = (12, 8),
  legend_pos: str = 'best',
  legend_cols: int = 1
):
  total = []
  limit = []
  colors = []

  classes = df['class'].unique()
  labels = [label_map[_class] for _class in classes]

  for i in range(n_folds):
    fold = df[df['fold'] == i]
    for _class in classes:
      objects = fold[fold['class'] == _class]
      total.append(objects['mag_r'].to_numpy())
      colors.append(color_map[_class])
    limit.append(fold['mag_r'].to_numpy())

  plt.figure(figsize=figsize)
  plt.hist(total, color=colors, stacked=True, bins=n_bins, label=labels)
  plt.hist(
    limit,
    stacked=True,
    color=('k',)*n_folds,
    bins=n_bins,
    histtype='bar',
    fill=False,
    linewidth=2
  )
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  if title is not None:
    plt.title(title)

  if label_map is not None:
    plt.legend(loc=legend_pos, ncol=legend_cols)
