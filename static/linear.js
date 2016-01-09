var linear = (typeof exports === "undefined")?(function linear() {}):(exports);
if(typeof global !== "undefined") { global.linear = linear; }

linear.vocab = [];
//Won't use this function, but it's here for an example of what a function should be like that I can then import into test2.js (or maybe into test3.js, renaming test2.js to be the page that relies on the Python backend for all its data and having test3.js be the page that takes the cooccurrence matrix from the Python backend and does the calculations on the client side with it)
linear.getVocab = function() {
    $.get("/vocab", function(data) {
        linear.vocab = data.vocab;
        return linear.vocab;
    });
}
