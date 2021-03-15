from minepy import MINE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_pearsons(data, target):

    """
    Calculate Pearson correlation between features and target
    :param data: DataFrame with data on molecules
    :param target: target name, str
    :return: DataFrame with Pearson correlations between each feature and target
    """

    corr_df = data.corr().loc[target].drop(target).reset_index()
    corr_df.columns = ['feature', 'correlation']
    corr_df = corr_df.sort_values(ascending=False, by='correlation')

    return corr_df


def calculate_mic(data, target):

    """
    Calculate MIC between features and target
    :param data: DataFrame with data on molecules
    :param target: target name, str
    :return: Series with MIC between each feature and target
    """

    X = data.drop(target, axis=1)
    y = data.loc[:, target]

    mine = MINE(alpha=0.6, c=15)
    mic_scores = {}
    for column in X.columns:
        mine.compute_score(X.loc[:, column].values, y.values)
        mic_scores[column] = mine.mic()
    mic_scores = pd.Series(mic_scores)

    return mic_scores


def plot_relationships(data, target):

    """
    Plot relationships between features and target
    :param data: DataFrame with data on molecules
    :param target: target name, str
    :return: plot with MIC and Pearson correlation values between each feature and target
    """

    corr_df = calculate_pearsons(data, target)
    mic_scores = calculate_mic(data, target)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    sns.barplot(ax=ax1,
                data=corr_df,
                x='feature',
                y='correlation',
                palette=sns.color_palette('YlGnBu_r',
                                          n_colors=corr_df.shape[0]))
    ax1.tick_params(labelrotation=45)
    ax1.set_xlabel('')
    ax1.set_title('Pearson correlation between features and target')

    sns.barplot(ax=ax2,
                data=mic_scores.sort_values(ascending=False, inplace=True),
                x=mic_scores.index,
                y=mic_scores.values,
                palette=sns.color_palette('YlOrRd_r', n_colors=mic_scores.shape[0]))
    ax2.tick_params(labelrotation=45)
    plt.ylabel('MIC')
    ax2.set_title('Maximum information coefficients for the features')
    ax2.set_xlabel('')
    plt.tight_layout()

    plt.show()
