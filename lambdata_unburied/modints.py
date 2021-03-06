from eli5 import show_weights
from eli5.sklearn import PermutationImportance
from IPython.display import display
from pdpbox.pdp import pdp_isolate, pdp_plot
from pdpbox.pdp import pdp_interact, pdp_interact_plot


#show feature weights
def permuter(model, X,y, **kwargs):
    """
    Uses eli5 package to plot permutation importance

    Scoring parameter keyword argument takes string arguments avilable here:
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    If no arguments are passed, defaults are:
    scoring = "accuracy",
    cv = "prefit",
    n_iter = 5
    """

    if 'scoring' in kwargs:
        scoring = kwargs['scoring']
    else:
        scoring = 'accuracy'
    if 'cv' in kwargs:
        cv = kwargs['cv']
    else:
        cv = 'prefit'
    if 'n_iter' in kwargs:
        n_iter = kwargs['n_iter']
    else:
        n_iter = 5
        
    perm= PermutationImportance(model,
                                scoring = scoring,
                                cv = cv,
                                n_iter = n_iter,
                                random_state = 42)

    #fit 
    perm.fit(X,y)

    #show weights based on feature names
    feature_names = X.columns.tolist()
    display(show_weights(perm, top=None, feature_names=feature_names))



def isolated(model, X, feature):
    """
    isolated pair dependancy plot
    """

    #instantiate and isolate variable
    isolated = pdp_isolate(model = model,
                        dataset = X,
                        model_features = X.columns,
                        feature = feature)
    #plot the variable
    pdp_plot(isolated, feature_name = feature)


def interaction(model, X, features, type = 'grid'):
    """
    plot interaction between features
    """
    #instantiate interaction vairable
    interaction = pdp_interact(model = model,
                            dataset = X,
                            model_features = X.columns,
                            features = features)
    #plot interactions
    pdp_interact_plot(interaction, plot_type = type, feature_names = features)