var linear = (typeof exports === "undefined")?(function linear() {}):(exports);
if(typeof global !== "undefined") { global.linear = linear; }

linear.vocab = [];
linear.cooccurrences = [];
linear.getData = function() {
    $.get("/linear-dataset", function(data) {
        linear.vocab = data.vocab;
        linear.cooccurrences = data.cooccurrences;
        console.log(linear.vocab);
    });
}
