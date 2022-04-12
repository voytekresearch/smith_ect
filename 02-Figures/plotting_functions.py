"""plotting functions for ECT"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


###########################################################################
################################ FUNCTIONS ################################
###########################################################################

def plot_feature_prepost(exp_df_mean, feature=''):
    """Creates a plot of a given feature from the dataframe,
       comparing its value before and after treatment

    Parameters
    ----------
    exp_df_mean: pandas DataFrame
        dataframe containing extractec features, averaged across electrodes
    feature: str
        feature of interest; 'alphas', 'thetas', 'chans_exps'

    Returns
    -------
    plot: matplotlib obj
        plot of feature pre/post treatment
    """
    
    pats = np.unique(exp_df_mean['patients'])

    # means of alphas across patients, sessions, channels
    means_pre = exp_df_mean[exp_df_mean['preposts']=='Pre'][feature].values
    means_post = exp_df_mean[exp_df_mean['preposts']=='Post'][feature].values

    sns.set_context('poster')

    plt.figure(figsize=(5,8))
    x1, x2 = 0.5, 1.5

    for pat in pats:
        sess = exp_df_mean[exp_df_mean['patients']==pat]['sessions'].unique()
        for ses in sess:
            point_pre = exp_df_mean[(exp_df_mean['patients']==pat) &
                                    (exp_df_mean['sessions']==ses) &
                                    (exp_df_mean['preposts']=='Pre')][feature].values
            point_post = exp_df_mean[(exp_df_mean['patients']==pat) &
                                     (exp_df_mean['sessions']==ses) &
                                     (exp_df_mean['preposts']=='Post')][feature].values
            xdat1= x1+np.random.normal(0, 0.025, 1)
            xdat2= x2+np.random.normal(0, 0.025, 1)

            # skip if pre&post not both present
            if point_pre.shape[0]==0 or point_post.shape[0]==0:
                pass
            else:
                plt.plot([xdat1,xdat2], [point_pre, point_post], color='k', alpha=0.05, lw=5)
                plt.scatter(xdat1, point_pre, color='teal', alpha=0.7)
                plt.scatter(xdat2, point_post, color='orange', alpha=0.7)

    plt.plot([x1-0.2, x1+0.2], [np.nanmean(means_pre), np.nanmean(means_pre)], lw=7, c='teal')
    plt.plot([x2-0.2, x2+0.2], [np.nanmean(means_post), np.nanmean(means_post)], lw=7, c='orange')
            
    plt.xlim([0, 2])

    plt.xticks([x1, x2], ["pre", "post"])

    if feature == 'delta_pows':
        ylabel = 'delta oscillation power'
    elif feature == 'theta_pows':
        ylabel = 'theta oscillation power'
    elif feature == 'chans_exps':
        ylabel = 'aperiodic exponent'
    else:
        ylabel = feature[:5]+' power'

    plt.ylabel(ylabel)

    plt.tight_layout()



def plot_feature_sessions(df, feature='', prepost=''):
    """plot feature over session from pre or post only

    Parameters
    ----------
    df: pandas DataFrame
        dataframe containing pre (or post) only measurements
    feature: str
        feature of interest; alphas, thetas, chans_exps
    prepost: str
        pre or post session recording included in plot; pre, post, diff

    Returns
    -------
    plot: matplotlib obj
        plot of single feature (Y) from pre or post only recording over sessions (X)
        """
    pats = np.unique(df['patients'])
    
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    
    #plt.figure(figsize=(12,8))
    cmap = [plt.cm.tab20b(i) for i in np.linspace(0, 1, len(pats))]  
    ax.set_prop_cycle('color', cmap)  

    #x_vals = [1, 4, 8, 12]
    # x_vals = 

    for pat in pats:
        ydat = []
        xdat = []
        sess = df[df['patients']==pat]['sessions'].unique()
        for ses in sess:
            x = ses
            # if ses == 1:
            #     x = x_vals[0]
            # if ses == 4:
            #     x = x_vals[1]
            # if ses == 8:
            #     x = x_vals[2]
            # if ses == 12:
            #     x = x_vals[3]
            # pre/post exponent vals for each session
            point_pre = df[(df['patients']==pat) &
                           (df['sessions']==ses) &
                           (df['preposts']=='Pre')][feature].values
            point_post = df[(df['patients']==pat) &
                            (df['sessions']==ses) &
                            (df['preposts']=='Post')][feature].values
            if point_pre.shape[0]==0:# point_post.shape[0]==0:
                pass
            else:
                if prepost == 'diff':
                    diff = point_post-point_pre
                    ydat.append(diff)
                    xdat.append(x)
                if prepost == 'pre':
                    ydat.append(point_pre)
                    xdat.append(x)
                if prepost == 'post':
                    ydat.append(point_post)
                    xdat.append(x)
        ax.plot(xdat, ydat,'o-', alpha=0.9, lw=5)

    ax.set_xlim(0.5,12.5)
    # ax.set_xlim([x_vals[0]-0.5, x_vals[-1]+0.5])
    #plt.ylim([-5, 5])

    #plt.xticks(x_vals, ["1", "4", "8", "12"])

    if feature == 'alphas':
        ylabel = 'Alpha CF'
    elif feature == 'thetas':
        ylabel = 'Theta CF'
    elif feature == 'chans_exps':
        ylabel = 'Aperiodic Exponent'
    else:
        ylabel = feature[:5]+' power'

    if prepost == 'diff':
        ax.set_ylabel('$\Delta$ ' + ylabel)
    else:
        ax.set_ylabel(ylabel)

    ax.set_xlabel('Session')

    plt.tight_layout()

    
def plot_spectra(avg_preSpec, avg_postSpec, loglog=True):
    """"plot average spectra for each patient and overall averages for pre and post
    
    Parameters
    ----------
    avg_preSpec : ndarray
    avg_postSpec : nd array
    loglog : bool
        plot in loglog scaling  if True, plot in semilog if False
    
    """
    
    # Calculate average Pre and Post Spectra

    # Trim zeros
    avg_preSpec = avg_preSpec[~np.all(avg_preSpec == 0, axis=1)]
    avg_postSpec = avg_postSpec[~np.all(avg_postSpec == 0, axis=1)]

    avg_avgPre = np.mean(avg_preSpec, axis=0)
    avg_avgPost = np.mean(avg_postSpec, axis=0)
    
    freqs = np.linspace(1,30,59)

    plt.figure(figsize=(10,8))
    sns.set_context('poster')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('power')

    for s in avg_postSpec:
        if s[1] == 0:
            pass
        else:
            plt.plot(freqs, s, color = 'orange', alpha=0.1, lw=4)
    for s in avg_preSpec:
        if s[1] == 0:
            pass
        else:
            plt.plot(freqs, s, color = 'teal', alpha=0.1, lw=4)

    plt.plot(freqs, avg_avgPre, color = 'teal', lw=10, label='pre')
    plt.plot(freqs, avg_avgPost, color = 'orange', lw=10, label='post')

    if loglog:
        plt.xscale('log')

    plt.legend(prop={'size': 25})
    sns.despine()

    plt.tight_layout()