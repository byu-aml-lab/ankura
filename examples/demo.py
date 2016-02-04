"""A demo of ankura functionality"""

import sklearn.decomposition
import ankura
import sys
import numpy
from numpy import linalg as LA


@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups-dataset.pickle')
def get_dataset():
    """Retrieves the 20 newsgroups dataset"""
    news_glob = '/local/jlund3/data/newsgroups/*/*'
    engl_stop = '/local/jlund3/data/stopwords/english.txt'
    news_stop = '/local/jlund3/data/stopwords/newsgroups.txt'
    name_stop = '/local/jlund3/data/stopwords/malenames.txt'
    pipeline = [(ankura.read_glob, news_glob, ankura.tokenize.news),
                (ankura.filter_stopwords, engl_stop),
                (ankura.filter_stopwords, news_stop),
                (ankura.combine_words, name_stop, '<name>'),
                (ankura.filter_rarewords, 100),
                (ankura.filter_commonwords, 1500)]
    dataset = ankura.run_pipeline(pipeline)
    return dataset



@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups-pca-anchor.pickle')
def pca_anchors():
    """Finds anchors using PCA"""
    dataset = get_dataset()
    pca = sklearn.decomposition.PCA(20)
    pca.fit(dataset.Q)
    return ankura.util.tuplize(pca.components_)


@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups-man-anchor.pickle')
def man_anchors():
    """Finds anchors using manifold gp"""
    dataset = get_dataset()
    Q = dataset.Q.copy()
    row_sums = Q.sum(1)
    for i in range(len(Q[:, 0])):
        Q[i, :] = Q[i, :] / float(row_sums[i])
    man = ManifoldGP(20, verbose=True, rescale=False)
    man.learn_landmarks(Q)
    return ankura.util.tuplize(man.landmarks)


def get_topics(dataset, anchors):
    """Wraps recover topics with memoize"""
    return ankura.recover_topics(dataset, anchors)


def print_summary(dataset, topics):
    """Prints a summary of the given topics"""
    for k in range(topics.shape[1]):
        summary = []
        for word in numpy.argsort(topics[:, k])[-10:][::-1]:
            summary.append(dataset.vocab[word])
        print(' '.join(summary))


def main():
    """Runs the demo"""
    dataset = get_dataset()
    anchors = man_anchors()

    def d(a, b):
        try:
            return numpy.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
        except:
            print(type(a), type(b))
            raise

    for an in anchors:
        i = min(range(dataset.vocab_size), key=lambda v: d(dataset.Q[v], an))
        print(dataset.vocab[i])
    topics = get_topics(dataset, anchors)
    print_summary(dataset, topics)


class ManifoldGP(object):
    ''' Landmarking manifolds with Gaussian processes
    and stochastic optimization '''
    def __init__(self, n_landmarks=100, batch_size=1000, n_steps=1000,
                 landmarks=None, init_lmk=None, proj=None, rescale=True,
                 random_state=None, verbose=True, **kwargs):
        ''' Gaussian processes manifold landmarking
        Arguments
        ---------
        n_landmarks : int
            Number of landmarks to learn (default 100)
        batch_size : int
            Batch size for stochastic gradient (default 1000)
        n_steps : int
            The number of gradient steps to take for each single landmark (
            default 1000)
        landmarks: ndarray or None
            Existing landmarks to begin with, should be in the shape of
            (n_existing_landmarks, n_feats), If None, landmarks will be learned
            from scratch
        init_lmk : callable or None
            A function to initialize the landmarks. If None, the default
            initialization on the unit sphere will be used
        proj : callable or None
            A projection function for ambient space that is not R^d. If None,
            the default projection to R^d will be used
        rescale : bool
            If true, the gradient is rescaled to have unit norm (default True)
            to prevent overshooting
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Stochastic gradient scheduling hyperparameters
        '''
        self.n_landmarks = n_landmarks
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.landmarks = landmarks
        self.rescale = rescale
        self.random_state = random_state
        self.verbose = verbose

        if callable(init_lmk):
            self.init_lmk = init_lmk
        else:
            self.init_lmk = _default_init

        if callable(proj):
            self.proj = proj
        else:
            self.proj = _do_nothing

        if type(self.random_state) is int:
            numpy.random.seed(self.random_state)
        elif self.random_state is not None:
            numpy.random.set_state(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.t0 = float(kwargs.get('t0', 0))
        self.gamma = float(kwargs.get('gamma', 0.5))

    def learn_landmarks(self, X, kern_width=None):
        ''' Fit the model to the data in X
        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.

        kern_width : float
            The kernel width. If None, set the kernel width to the sum of the
            per-dimensional empirical variance

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        if kern_width is None:
            self.kern_width = numpy.var(X, axis=0).sum()
        else:
            self.kern_width = kern_width

        for l in range(self.n_landmarks):
            if self.verbose:
                print("Learning landmark %i:" % l)
                sys.stdout.flush()
            lmk = self._learn_single_landmark(X)
            if self.landmarks is None:
                self.landmarks = lmk[numpy.newaxis, :]
            else:
                self.landmarks = numpy.vstack((self.landmarks, lmk))
        return self

    def _learn_single_landmark(self, X):
        ''' learn a single landmark '''
        n_samples, n_feats = X.shape
        lmk = self.init_lmk(n_feats)
        for i in range(1, self.n_steps + 1):
            idx = numpy.random.choice(n_samples, size=self.batch_size)
            lmk = self._grad_step(X[idx], lmk, i)
            if self.verbose and i % 50 == 0:
                sys.stdout.write('\rProgress: %d/%d' % (i, self.n_steps))
                sys.stdout.flush()
        if self.verbose:
            sys.stdout.write('\n')
        return lmk

    def _grad_step(self, X_batch, lmk, step):
        ''' take one stochastic gradient step '''
        phi = numpy.exp(-(2 - 2 * X_batch.dot(lmk)) / self.kern_width)
        if self.landmarks is None:
            M2phi = phi
        else:
            K = numpy.exp(-(2 - 2 * X_batch.dot(self.landmarks.T))
                       / self.kern_width)
            M2phi = phi - K.dot(LA.lstsq(K.T.dot(K), K.T.dot(phi))[0])
        rho = (self.t0 + step)**(-self.gamma)
        grad_lmk = -4. / self.kern_width * (lmk * phi.dot(M2phi) -
                                            X_batch.T.dot(M2phi * phi))
        if self.rescale:
            grad_lmk /= LA.norm(grad_lmk)
        lmk += rho * grad_lmk
        return self.proj(lmk)


# helper functions to initialize landmarks and projection #
def _do_nothing(x):
    ''' no projection, for R^d ambient space '''
    return x


def proj_pos(x):
    ''' project to the positive orthant '''
    x[x < 0] = 0
    return x


def proj_sph_pos(x):
    ''' project to the intersection of the unit sphere with positive orthant '''
    x = proj_pos(x)
    return x/LA.norm(x)


def _default_init(n_feats):
    ''' default initialization on the intersection of the unit sphere with
    positive orthant (if you know what you are doing, you should probably not
    use it) '''
    lmk = proj_sph_pos(numpy.ones(n_feats))
    return lmk


if __name__ == '__main__':
    main()


