from __future__ import division

import warnings
import numpy as np
from scipy import sparse
from math import sqrt

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms

from seqdesign.text_histogram import MVSD


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('This LinearSVC instance is not fitted yet'...)
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    Returns
    -------
    None
    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    n_samples = X.shape[0]
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    for i in range(n_samples):
        row = np.zeros(X.shape[1])
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row


def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    new_node1 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_node2 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    dist = euclidean_distances(
        node.centroids_, Y_norm_squared=node.squared_norm_, squared=True)
    n_clusters = dist.shape[0]

    farthest_idx = np.unravel_index(
        dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[[farthest_idx]]

    node1_closer = node1_dist < node2_dist
    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _CFNode(object):
    """Each node in a CFTree is called a CFNode.
    The CFNode can have a maximum of branching_factor
    number of CFSubclusters.
    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a CFSubcluster.
    branching_factor : int
        Maximum number of CF subclusters in each node.
    is_leaf : bool
        We need to know if the CFNode is a leaf or not, in order to
        retrieve the final subclusters.
    n_features : int
        The number of features.
    Attributes
    ----------
    subclusters_ : array-like
        list of subclusters for a particular CFNode.
    prev_leaf_ : _CFNode
        prev_leaf. Useful only if is_leaf is True.
    next_leaf_ : _CFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.
    init_centroids_ : ndarray, shape (branching_factor + 1, n_features)
        manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .
    init_sq_norm_ : ndarray, shape (branching_factor + 1,)
        manipulate init_sq_norm_ throughout. similar to ``init_centroids_``.
    centroids_ : ndarray
        view of ``init_centroids_``.
    squared_norm_ : ndarray
        view of ``init_sq_norm_``.
    """
    def __init__(self, threshold, branching_factor, is_leaf, n_features):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features

        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features))
        self.init_sq_norm_ = np.zeros((branching_factor + 1))
        self.squared_norm_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_

        # Keep centroids and squared norm as views. In this way
        # if we change init_centroids and init_sq_norm_, it is
        # sufficient,
        self.centroids_ = self.init_centroids_[:n_samples + 1, :]
        self.squared_norm_ = self.init_sq_norm_[:n_samples + 1]

    def update_split_subclusters(self, subcluster,
                                 new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        self.append_subcluster(new_subcluster2)

    def insert_cf_subcluster(self, subcluster):
        """Insert a new subcluster into the node."""
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold = self.threshold
        branching_factor = self.branching_factor
        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        dist_matrix = np.dot(self.centroids_, subcluster.centroid_)
        dist_matrix *= -2.
        dist_matrix += self.squared_norm_
        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # If the subcluster has a child, we need a recursive strategy.
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(
                subcluster)

            if not split_child:
                # If it is determined that the child need not be split, we
                # can just update the closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = \
                    self.subclusters_[closest_index].centroid_
                self.init_sq_norm_[closest_index] = \
                    self.subclusters_[closest_index].sq_norm_
                return False

            # things not too good. we need to redistribute the subclusters in
            # our child node, and add a new subcluster in the parent
            # subcluster to accommodate the new child.
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_, threshold, branching_factor)
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2)

                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # good to go!
        else:
            merged = closest_subcluster.merge_subcluster(
                subcluster, self.threshold)
            if merged:
                self.init_centroids_[closest_index] = \
                    closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = \
                    closest_subcluster.sq_norm_
                return False

            # not close to any other subclusters, and we still
            # have space, so add.
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False

            # We do not have enough space nor is it closer to an
            # other subcluster. We need to split.
            else:
                self.append_subcluster(subcluster)
                return True


class _CFSubcluster(object):
    """Each subcluster in a CFNode is called a CFSubcluster.
    A CFSubcluster can have a CFNode has its child.
    Parameters
    ----------
    linear_sum : ndarray, shape (n_features,), optional
        Sample. This is kept optional to allow initialization of empty
        subclusters.
    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.
    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.
    squared_sum_ : float
        Sum of the squared l2 norms of all samples belonging to a subcluster.
    centroid_ : ndarray
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``CFNode.centroids_`` is called.
    child_ : _CFNode
        Child Node of the subcluster. Once a given _CFNode is set as the child
        of the _CFNode, it is set to ``self.child_``.
    sq_norm_ : ndarray
        Squared norm of the subcluster. Used to prevent recomputing when
        pairwise minimum distances are computed.
    """
    def __init__(self, linear_sum=None):
        if linear_sum is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.linear_sum_ = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.squared_sum_ = self.sq_norm_ = np.dot(
                self.linear_sum_, self.linear_sum_)
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = (1 / new_n) * new_ls
        new_norm = np.dot(new_centroid, new_centroid)
        dot_product = (-2 * new_n) * new_norm
        sq_radius = (new_ss + dot_product) / new_n + new_norm
        if sq_radius <= threshold ** 2:
            (self.n_samples_, self.linear_sum_, self.squared_sum_,
             self.centroid_, self.sq_norm_) = \
                new_n, new_ls, new_ss, new_centroid, new_norm
            return True
        return False

    @property
    def radius(self):
        """Return radius of the subcluster"""
        dot_product = -2 * np.dot(self.linear_sum_, self.centroid_)
        return sqrt(
            ((self.squared_sum_ + dot_product) / self.n_samples_) +
            self.sq_norm_)


class BirchIter:
    """Implements the Birch clustering algorithm.
    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.
    Read more in the :ref:`User Guide <birch>`.
    Parameters
    ----------
    threshold : float, default 0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.
    branching_factor : int, default 50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.
    n_clusters : int, instance of sklearn.cluster model, default 3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.
        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.
        - `sklearn.cluster` Estimator : If a model is provided, the model is
          fit treating the subclusters as new samples and the initial data is
          mapped to the label of the closest subcluster.
        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.
    compute_labels : bool, default True
        Whether or not to compute labels for each fit.
    copy : bool, default True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.
    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.
    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.
    subcluster_centers_ : ndarray,
        Centroids of all subclusters read directly from the leaves.
    subcluster_labels_ : ndarray,
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.
    Examples
    --------
    >>> from sklearn.cluster import Birch
    >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    >>> brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5,
    ... compute_labels=True)
    >>> brc.fit(X)
    Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None,
       threshold=0.5)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])
    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf
    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch
    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, squared sum and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.
    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, squared sum and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.
    """

    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3,
                 compute_labels=True, copy=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy

    # Pass in the custom data helper function that will process the data
    #     instead of loading it all in memory at once
    def fit(self, data_helper):

        threshold = self.threshold
        branching_factor = self.branching_factor

        if branching_factor <= 1:
            raise ValueError("Branching_factor should be greater than one.")

        n_features = data_helper.kmer_feat_num

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        partial_fit = True
        #has_root = getattr(self, 'root_', None)
        #has_root =
        #if getattr(self, 'fit_') or (partial_fit and not has_root):


        # The first root is the leaf. Manipulate this object throughout.
        self.root_ = _CFNode(threshold, branching_factor, is_leaf=True,
                             n_features=n_features)

        # To enable getting back subclusters.
        self.dummy_leaf_ = _CFNode(threshold, branching_factor,
                                   is_leaf=True, n_features=n_features)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_

        counter = 0
        # Iterate through all the sequences
        for name,kmer_data_list in data_helper.seq_name_to_kmer_data_lists.items():
            sample = n_features * [0.]
            for idx_kmer,count in kmer_data_list:
                sample[idx_kmer] = count
            sample = np.asarray(sample)
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor)
                del self.root_
                self.root_ = _CFNode(threshold, branching_factor,
                                     is_leaf=False,
                                     n_features=n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)
            if counter % 1000 == 0:
                print(counter, flush=True)
            counter += 1

        centroids = np.concatenate([
            leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids
        self._subcluster_norms = row_norms(self.subcluster_centers_, squared=True)

        self.num_clusters = self.subcluster_centers_.shape[0]

        #self._global_clustering(X)
        return self

    def _get_leaves(self):
        """
        Retrieve the leaves of the CF Node.
        Returns
        -------
        leaves : array-like
            List of the leaf nodes.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def partial_fit(self, X=None, y=None):
        """
        Online learning. Prevents rebuilding of CFTree from scratch.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features), None
            Input data. If X is not provided, only the global clustering
            step is done.
        y : Ignored
        """
        self.partial_fit_, self.fit_ = True, False
        if X is None:
            # Perform just the final global clustering step.
            self._global_clustering()
            return self
        else:
            self._check_fit(X)
            return self._fit(X)

    def _check_fit(self, X):
        is_fitted = hasattr(self, 'subcluster_centers_')

        # Called by partial_fit, before fitting.
        has_partial_fit = hasattr(self, 'partial_fit_')

        # Should raise an error if one does not fit before predicting.
        if not (is_fitted or has_partial_fit):
            raise NotFittedError("Fit training data before predicting")

        if is_fitted and X.shape[1] != self.subcluster_centers_.shape[1]:
            raise ValueError(
                "Training data and predicted data do "
                "not have same number of features.")

    def predict(self, data_helper, minibatch_size=1000, output_name='output.csv'):
        """
        Predict data using the ``centroids_`` of subclusters.
        Avoid computation of the row norms of X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        labels : ndarray, shape(n_samples)
            Labelled data.
        """
        OUTPUT = open(output_name, 'w')

        completed = 0
        for i in range(0,len(data_helper.seq_name_list),minibatch_size):

            batch_names = data_helper.seq_name_list[i:i+minibatch_size]

            minibatch_kmer_data = np.zeros((len(batch_names),data_helper.kmer_feat_num))

            for j,name in enumerate(batch_names):
                kmer_data_list = data_helper.seq_name_to_kmer_data_lists[name]
                for k_kmer,count in kmer_data_list:
                    minibatch_kmer_data[j,k_kmer] = count

            reduced_distance = np.dot(minibatch_kmer_data, self.subcluster_centers_.T)
            reduced_distance *= -2
            reduced_distance += self._subcluster_norms

            cluster_num = np.argmin(reduced_distance, axis=1)
            cluster_dist = np.min(reduced_distance, axis=1)

            for j, name in enumerate(batch_names):
                OUTPUT.write(f'{name},{cluster_num[j]},{cluster_dist[j]}\n')

            completed += len(batch_names)
            print(completed, flush=True)

        OUTPUT.close()

    def transform(self, X):
        """
        Transform X into subcluster centroids dimension.
        Each dimension represents the distance from the sample point to each
        cluster centroid.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        X_trans : {array-like, sparse matrix}, shape (n_samples, n_clusters)
            Transformed data.
        """
        check_is_fitted(self, 'subcluster_centers_')
        return euclidean_distances(X, self.subcluster_centers_)


class NanobodyDataBirchCluster:
    def __init__(
            self, input_filename='',r_seed=42, n_optimize=100,
            minibatch_size=150, contin_feat_num=4
    ):

        np.random.seed(r_seed)

        self.minibatch_size = minibatch_size
        self.contin_feat_num = contin_feat_num
        self.n_optimize = n_optimize
        self.seq_name_to_cluster_list = {}

        self.length_mvsd = MVSD()
        self.hydro7_mvsd = MVSD()
        self.pi_mvsd = MVSD()
        self.mw_mvsd = MVSD()

        cdr3_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        kmer_to_idx = {}
        counter = 0
        kmer_list = [aa for aa in cdr3_alphabet]
        for aa in cdr3_alphabet:
            for bb in cdr3_alphabet:
                kmer_list.append(aa+bb)
                for cc in cdr3_alphabet:
                    kmer_list.append(aa+bb+cc)

        kmer_to_idx = {aa:i for i,aa in enumerate(kmer_list)}

        with open(self.input_filename, 'r') as INPUT:
            header = next(INPUT)
            for line in INPUT:
                line = line.rstrip()
                line_list = line.split(',')
                name = line_list.pop(0)
                self.length_mvsd.add(float(line_list.pop(0)))
                hydro_ph2 = line_list.pop(0)  # this one isn't used
                self.hydro7_mvsd.add(float(line_list.pop(0)))
                self.pi_mvsd.add(float(line_list.pop(0)))
                self.mw_mvsd.add(float(line_list.pop(0)))

        self.contin_feat_num = contin_feat_num
        self.kmer_feat_num = len(kmer_list)

        self.input_filename = input_filename
        self.seq_name_list = []
        self.seq_name_to_continuous_feat = {}
        self.seq_name_to_kmer_data_lists = {}
        self.seq_name_to_number = {}
        with open(self.input_filename, 'r') as INPUT:
            header = next(INPUT)
            for i, line in enumerate(INPUT):
                line = line.rstrip()
                line_list = line.split(',')
                name = line_list.pop(0)
                length = (float(line_list.pop(0)) - self.length_mvsd.mean()) / self.length_mvsd.sd()
                hydro_ph2 = line_list.pop(0)  # this one isn't used
                hydro_ph7 = (float(line_list.pop(0)) - self.hydro7_mvsd.mean()) / self.hydro7_mvsd.sd()
                pI = (float(line_list.pop(0)) - self.pi_mvsd.mean()) / self.pi_mvsd.sd()
                mw = (float(line_list.pop(0)) - self.mw_mvsd.mean()) / self.mw_mvsd.sd()

                kmer_data_list = line_list
                kmer_data_list = [val.split(':') for val in kmer_data_list]
                final_kmer_data_list = []

                # calculate the norm first and save that so I don't have to do it redundantly
                norm_val = sqrt(sum(int(count) ** 2 for kmer, count in kmer_data_list))

                for kmer,count in kmer_data_list:
                    final_kmer_data_list.append((kmer_to_idx[kmer],float(count)/norm_val))

                self.seq_name_to_kmer_data_lists[name] = final_kmer_data_list
                self.seq_name_to_continuous_feat[name] = [length,hydro_ph7,pI,mw]
                self.seq_name_list.append(name)
                self.seq_name_to_number[name] = i-1
        INPUT.close()

        # first shuffle all the names that will go in the library
        np.random.shuffle(self.seq_name_list)

        # then pick a set of candidates to start with
        self.lib_names = self.seq_name_list[:n_optimize]

        # the rest are names that are new potential candidates
        self.option_pool_names = self.seq_name_list[n_optimize:]

        self.kmer_data_arr = np.zeros((n_optimize, len(kmer_list)))
        self.continuous_data_arr = np.zeros((n_optimize, contin_feat_num))

        for i, name in enumerate(self.lib_names):
            kmer_data_list = self.seq_name_to_kmer_data_lists[name]
            contin_data_list = self.seq_name_to_continuous_feat[name]
            for j, val in enumerate(contin_data_list):
                self.continuous_data_arr[i, j] = val
            for j_kmer, count in kmer_data_list:
                self.kmer_data_arr[i, j_kmer] = count

        # then initialize the minibatch data to zeros for later generation
        #  Include 1 more to include the thing we are replacing
        # I don't want to be reinitializing this array over and over again in mem
        self.minibatch_kmer_data_arr = np.zeros((minibatch_size + 1, len(kmer_list)))
        self.minibatch_continuous_data_arr = np.zeros((minibatch_size + 1, contin_feat_num))
        self.library_mask = np.ones((n_optimize,))

        # self.seq_name_to_seq = {}
        # INPUT = open('nanobody_id80_test_nanobodies_seqs.csv', 'r')
        # for line in INPUT:
        #     line = line.rstrip()
        #     if line[0] == '>':
        #         name = line
        #     else:
        #         self.seq_name_to_seq[name] = line

    def gen_minibatch(self, idx_replace):

        self.minibatch_names = self.lib_names[idx_replace]

        # Make these back to zero
        self.minibatch_kmer_data_arr *= 0.
        self.minibatch_continuous_data_arr *= 0.

        self.library_mask *= 0.
        self.library_mask += 1.
        self.library_mask[idx_replace] = 0.

        # Make the first value in these arrays the thing that may be replaced
        self.minibatch_kmer_data_arr[0] = self.kmer_data_arr[idx_replace]
        self.minibatch_continuous_data_arr[0] = self.continuous_data_arr[idx_replace]

        new_minibatch_names = np.random.choice(self.option_pool_names,self.minibatch_size).tolist()

        self.minibatch_names += new_minibatch_names

        for i,name in enumerate(new_minibatch_names):
            kmer_data_list = self.seq_name_to_kmer_data_lists[name]
            contin_data_list = self.seq_name_to_continuous_feat[name]
            for j,val in enumerate(contin_data_list):
                self.minibatch_continuous_data_arr[i+1,j] = val
            for j_kmer,count in kmer_data_list:
                self.minibatch_kmer_data_arr[i+1,j_kmer] = count
