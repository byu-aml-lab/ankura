from __future__ import division

def pR2(model, words, labels, labelmean):
    total_ss = 0.0
    residual_ss = 0.0
    for (text, label) in zip(words, labels):
        prediction = model.predict(text)
        residual_ss += (label - prediction)**2
        total_ss += (label - labelmean)**2
    return 1.0 - (residual_ss / total_ss)
