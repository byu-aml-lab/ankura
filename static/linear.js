var linear = (typeof exports === "undefined")?(function linear() {}):(exports);
if(typeof global !== "undefined") { global.linear = linear; }

linear.vocab = [];
linear.getVocab = function() {
    $.get("/vocab", function(data) {
        linear.vocab = data.vocab;
        return linear.vocab;
    });
}
