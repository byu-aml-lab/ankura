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

linear.sumRow = function sumRow(row) {
    var rowLen = row.length;
    var sum = 0;
    for (var i = 0; i < rowLen; i++) {
        sum += row[i];
    }
    return sum;
}

//Constructs basis vectors from a list of anchor indices
linear.anchorVectors = function anchorVectors(cooccMatrix, anchors) {
    var basis = linear.matrixZeroes(anchors.length, cooccMatrix[0].length);
    var cooccLength = cooccMatrix.length;
    for (var i = 0; i < cooccLength; i++) {
        basis[i] = (linear.sumRow(cooccMatrix[i])/cooccMatrix[i].length);
    }
    return basis;
}

//Recovers topics given a set of anchors and cooccurrences matrix
linear.recoverTopics = function recoverTopics(cooccMatrix, anchors) {
    //We don't want to modify the original cooccurrences matrix
    var Q = linear.deepCloneMatrix(cooccMatrix);

    var V = cooccMatrix.length;
    var K = anchors.length;
    var A = linear.matrixZeroes(V, K);

    //Create a diagonal matrix, where the ith entry of the ith row in
    //  P_w is the sum of the row in Q.
    var P_w = numeric.diag(linear.sumMatrixRows(Q));
    var P_wLength = P_w.length;
    //This check was in the Python code, not sure why.
    for (var i = 0; i < P_wLength; i++) {
        if (isNaN(P_w[i][i])) {
            P_w[i][i] = Math.pow(1, -16);
        }
    }
    //Normalize the rows of Q to get Q_prime
    Q = linear.normalizeMatrixRows(Q);

    var anchorCopy = linear.deepCloneMatrix(anchors);
    console.log(anchorCopy);
    //Compute normalized anchors X, and precompute X * X.T
    anchors = linear.anchorVectors(cooccMatrix, anchors);
    console.log(anchors);
}
