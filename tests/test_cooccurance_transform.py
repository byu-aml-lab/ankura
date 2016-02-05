import ankura
import pytest
import scipy
import numpy

old_matrix = [[0, 1, 0],
              [0, 0, 0],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]]

old_vocab = ["fox", "badger", "walrus", "ringle", "bingle"]
old_docs = ["doc1", "doc2", "doc3"]
dataset_matrix1 = scipy.sparse.lil_matrix(old_matrix)
test_dataset1 = ankura.pipeline.Dataset(dataset_matrix1, old_vocab, old_docs)
expected_matrix1= [[0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 1],
                   [0, 0, 1]]
expected_matrix1 = scipy.sparse.lil_matrix(expected_matrix1)
expected_vocab1 = ["fox-walrus", "fox-bingle",
                    "walrus-ringle", "walrus-bingle", "ringle-bingle"]
expected_docs = ["doc1", "doc2", "doc3"]
expected_dataset1 = ankura.pipeline.Dataset(expected_matrix1, expected_vocab1,
                                           expected_docs)
old_matrix1 = [[.00, .11, .22, .01],
               [.23, .84, .23, .02],
               [.001, .123, .233, .03],
               [1.23, 2.33, .23, .04],
               [.009, .073, .233, .05]]
old_matrix1 = scipy.sparse.lil_matrix(old_matrix1)
old_vocab1 = ["skippy", "skippi", "skyppi", "skyppy", "skeppe"]
old_docs1 = ["doc1", "doc2", "doc3", "doc4"] 
test_dataset2 = ankura.pipeline.Dataset(old_matrix1, old_vocab1, old_docs1)
expected_matrix2 = [[.00, .11, .22, .01],
                    [.00, .11, .22, .01],
                    [.00, .11, .22, .01],
                    [.00, .073, .22, .01],
                    [.001, .123, .23, .02],
                    [.23, .84, .23, .02],
                    [.009, .073, .23, .02],
                    [.001, .123, .23, .03],
                    [.001, .073, .233, .03],
                    [.009, .073, .233, .04]]
expected_matrix2 = scipy.sparse.lil_matrix(expected_matrix2)
expected_vocab2 = ["skippy-skippi", "skippy-skyppi", "skippy-skyppy",
                   "skippy-skeppe", "skippi-skyppi", "skippi-skyppy", "skippi-skeppe",
                   "skyppi-skyppy", "skyppi-skeppe", "skyppy-skeppe"]
expected_dataset2 = ankura.pipeline.Dataset(expected_matrix2, expected_vocab2,
                                             old_docs1)

old_matrix3 = [[-2, 2],
               [1, -1],
               [-3, 3],
               [4, -2],
               [-6, 5],
               [0, -2]]
old_matrix3 = scipy.sparse.lil_matrix(old_matrix3)
old_vocab3 = ["dill", "will", "jill", "kill", "hill", "bill"]
old_docs3 = ["doc1", "doc2"]
test_dataset3 = ankura.pipeline.Dataset(old_matrix3, old_vocab3, old_docs3)

expected_matrix3 = [[-2, -1],
                    [-3, 2],
                    [-2, -2],
                    [-6, 2],
                    [-2, -2],
                    [-3, -1],
                    [1, -2],
                    [-6, -1],
                    [0, -2],
                    [-3, -2],
                    [-6, 3],
                    [-3, -2],
                    [-6, -2],
                    [0, -2],
                    [-6, -2]]
expected_matrix3 = scipy.sparse.lil_matrix(expected_matrix3)
expected_vocab3 = ["dill-will", "dill-jill", "dill-kill", "dill-hill",
                   "dill-bill", "will-jill", "will-kill", "will-hill", "will-bill", "jill-kill",
                   "jill-hill", "jill-bill", "kill-hill", "kill-bill", "hill-bill"]
expected_dataset3 = ankura.pipeline.Dataset(expected_matrix3, expected_vocab3,
                                            old_docs3)

old_matrix4 = [[]]
old_matrix4 = scipy.sparse.lil_matrix(old_matrix4) 
old_vocab4 = []
old_docs4 = []
test_dataset4 = ankura.pipeline.Dataset(old_matrix4, old_vocab4, old_docs4)

expected_matrix4 = [[]]
expected_matrix4 = scipy.sparse.lil_matrix(expected_matrix4)
expected_dataset4 = ankura.pipeline.Dataset(expected_matrix4, old_vocab4,
                                            old_docs4)

old_matrix5 = [[5]]
old_matrix5 = scipy.sparse.lil_matrix(old_matrix5)
old_vocab5 = ["lizard"]
old_docs5 = ["doc1"]
test_dataset5 = ankura.pipeline.Dataset(old_matrix5, old_vocab5, old_docs5)

expected_matrix5 = []
expected_vocab5 = []
expected_docs5 = []
expected_matrix5 = scipy.sparse.lil_matrix(expected_matrix5)
expected_dataset5 = ankura.pipeline.Dataset(expected_matrix5, expected_vocab5,
                                            expected_docs5)
old_matrix6 = [[1],
               [2],
               [0],
               [4]]
old_matrix6 = scipy.sparse.lil_matrix(old_matrix6)
old_vocab6 = ["pirate", "yellow", "cheese", "grubby"]
old_docs6 = ["doc1"]
test_dataset6 = ankura.pipeline.Dataset(old_matrix6, old_vocab6, old_docs6)

expected_matrix6 = [[1],
                    [1],
                    [2]]
expected_matrix6 = scipy.sparse.lil_matrix(expected_matrix6)
expected_vocab6 = ["pirate-yellow", "pirate-grubby",
                   "yellow-grubby"]
expected_dataset6 = ankura.pipeline.Dataset(expected_matrix6, expected_vocab6,
                                            old_docs6)
def test_dataset_transform1():
    new_dataset1 = ankura.pipeline.get_word_cooccurrences(test_dataset1)
    assert dataset_equals(new_dataset1, expected_dataset1)

def test_dataset_transform2():
    new_dataset2 = ankura.pipeline.get_word_cooccurrences(test_dataset2)
    assert dataset_equals(new_dataset2, expected_dataset2)

def test_dataset_transform3():
    new_dataset3 = ankura.pipeline.get_word_cooccurrences(test_dataset3)
    assert dataset_equals(new_dataset3, expected_dataset3)

def test_dataset_transform4():
    new_dataset4 = ankura.pipeline.get_word_cooccurrences(test_dataset4)
    assert dataset_equals(new_dataset4, expected_dataset4)

def test_dataset_transform5():
    new_dataset5 = ankura.pipeline.get_word_cooccurrences(test_dataset5)
    assert dataset_equals(new_dataset5, expected_dataset5)

def test_dataset_transform6():
    new_dataset6 = ankura.pipeline.get_word_cooccurrences(test_dataset6)
    assert dataset_equals(new_dataset6, expected_dataset6)

def dataset_equals(dataset1, dataset2):
    print(dataset1.vocab)
    print(dataset2.vocab)
    print(dataset1.M.toarray())
    print(dataset2.M.toarray())
    print(dataset1.titles)
    print(dataset2.titles)
    if dataset1.vocab != dataset2.vocab:
        return False
    elif numpy.array_equal(dataset1.M, dataset2.M):
        return False
    elif dataset1.titles != dataset2.titles:
        return False
    else:
        return True
test_dataset_transform1()
test_dataset_transform2()
test_dataset_transform3()
test_dataset_transform4()
test_dataset_transform5()
test_dataset_transform6()
