var linear = (typeof exports === "undefined")?(function linear() {}):(exports);
if(typeof global !== "undefined") { global.linear = linear; }

//When the term "matrix" is used, it means an array of arrays.

//Returns an array with the sums of each row of a matrix
linear.sumMatrixRows = function sumMatrixRows(matrix) {
    var colLength = matrix.length;
    var sumOfRows = [];
    for (var i = 0; i < colLength; i++) {
        var rowLength = matrix[i].length;
        var sum = 0;
        for (var j = 0; j < rowLength; j++) {
            sum += matrix[i][j];
        }
        sumOfRows.push(sum);
    }
    return sumOfRows;
}

//Creates a matrix of the specified number of rows and columns full of zeroes
linear.matrixZeroes = function matrixZeroes(rows, cols) {
    var arr = [];
    for (var i = 0; i < rows; i++) {
        arr[i] = [];
        for (var j = 0; j < cols; j++) {
            arr[i][j] = 0;
        }
    }
    return arr;
}

//Creates a deep copy of a matrix
linear.deepCloneMatrix = function deepCloneMatrix(arr) {
  var len = arr.length;
  var newArr = new Array(len);
  for (var i=0; i<len; i++) {
    if (Array.isArray(arr[i])) {
      newArr[i] = deepCloneMatrix(arr[i]);
    }
    else {
      newArr[i] = arr[i];
    }
  }
  return newArr;
}

//Normalizes the rows of the matrix passed into it
linear.normalizeMatrixRows = function normalizeMatrixRows(matrix) {
    var normalizedMatrix = matrix;
    var matrixLength = normalizedMatrix.length;
    for (var i = 0; i < matrixLength; i++) {
        var rowLength = normalizedMatrix[i].length;
        //Need to sum the row first to normalize it correctly.
        var rowSum = 0;
        for (var j = 0; j < rowLength; j++) {
            rowSum += normalizedMatrix[i][j];
        }
        //Now we normalize the row values
        for (var j = 0; j < rowLength; j++) {
            normalizedMatrix[i][j] = (normalizedMatrix[i][j]/rowSum);
        }
    }
    return normalizedMatrix;
}

//Sums an array of numbers
linear.sumRow = function sumRow(row) {
    var sum = 0;
    for (var i = 0; i < row.length; i++) {
        sum += row[i];
    }
    return sum;
}

//Creates a single basis vector given the cooccurrences matrix and an anchor
linear.createBasisVector = function createBasisVector(cooccMatrix, anchor) {
    var basisVector = [];
    for (var i = 0; i < cooccMatrix.length; i++) {
        var sum = 0;
        for (var j = 0; j < anchor.length; j++) {
            sum += cooccMatrix[anchor[j]][i];
        }
        basisVector[i] = sum / anchor.length;
    }
    return basisVector;
}

//Constructs basis vectors from a list of anchor indices
linear.anchorVectors = function anchorVectors(cooccMatrix, anchors, vocab) {
    var basis = [];
    for (var i = 0; i < anchors.length; i++) {
        var anchor = [];
        for (var j = 0; j < anchors[i].length; j++) {
            anchor.push(vocab.indexOf(anchors[i][j]));
        }
        basis[i] = linear.createBasisVector(cooccMatrix, anchor, vocab);
    }
    return basis;
}

//Computes the X matrix, which is just a row-normalized basis
linear.computeX = function computeX(anchors) {
    var X = linear.deepCloneMatrix(anchors);
    for (var i = 0; i < X.length; i++) {
        var rowSum = linear.sumRow(X[i]);
        for (var j = 0; j < X[i].length; j++) {
            X[i][j] = (X[i][j]/rowSum);
        }
    }
    return X;
}

linear.exponentiatedGradient = function exponentiatedGradient(Y, X, XX, epsilon) {

}

//Recovers topics given a set of anchors (as words) and a cooccurrences matrix
linear.recoverTopics = function recoverTopics(cooccMatrix, anchors, vocab) {
    //We don't want to modify the original cooccurrences matrix
    var Q = linear.deepCloneMatrix(cooccMatrix);

    var V = cooccMatrix.length;
    var K = anchors.length;
    var A = linear.matrixZeroes(V, K);

    //Create a diagonal matrix, where the ith entry of the ith row in
    //  P_w is the sum of the row in Q.
    var P_w = numeric.diag(linear.sumMatrixRows(Q));
    //This check was in the Python code, not sure why.
    for (var i = 0; i < P_w.length; i++) {
        if (isNaN(P_w[i][i])) {
            //Put in a really small number to avoid division by zero?
            P_w[i][i] = Math.pow(10, -16);
        }
    }
    //Normalize the rows of Q to get Q_prime
    Q = linear.normalizeMatrixRows(Q);

    //Compute normalized anchors X, and precompute X * X.T
    anchors = linear.anchorVectors(cooccMatrix, anchors, vocab);
    var X = linear.computeX(anchors);
    var X_T = linear.deepCloneMatrix(X);
    X_T = numeric.transpose(X_T);
    var XX = numeric.dot(X, X_T);

    //Do exponentiated gradient descent
    var epsilon = Math.pow(10, -7);
    for (var i = 0; i < V; i++) {
        //Y = cooccMatrix[i];
        var alpha = linear.exponentiatedGradient(cooccMatrix[i],
                                                    X, XX, epsilon);
        //if numpy.isnan(alpha).any():?????
    }
}
