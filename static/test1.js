angular.module('anchorApp', [])
  .controller('anchorController', function($scope) {
    var ctrl = this;
    $scope.getTopics() = function() {
      ctrl.anchors = [];
      $.get("/topics", function(data) {
        for (var i = 0; i < data["anchors"].length; i++) {
          var anchor = JSON.stringify(data["anchors"][i]);
          var topic = JSON.stringify(data["topics"][i]);
          console.log(anchor);
          console.log(topic);
          ctrl.anchors.push({"anchor":anchor, "topic":topic});
        }
        console.log(ctrl.anchors);
        return ctrl.anchors;
      });
    }
  });

function update(anchors) {
  console.log(anchors);
  $("#topics").html(anchors);
}
/*
function update(data) {
  console.log(data);
  $("#topics").html(
    JSON.stringify(data["topics"])
      .replace(/\],/g, "\n")
      .replace(/["\[\]]/g, "")
  )
}

$(document).ready(function() {
  $.get("/topics", function(data) {
    $("#anchors").val(JSON.stringify(data["topics"]));
    console.log(JSON.stringify(data["anchors"].join()));
    update(data);
  });
  $("#anchor-form").submit(function(e) {
    e.preventDefault();
    var anchorArr = $("#anchors").val().split("\n");
    console.log(anchorArr);
    for (var i = 0; i < anchorArr.length; i++) {
      var anchor = anchorArr[i];
      anchorArr[i] = anchor.split(",");
    };
    console.log(anchorArr);
    console.log($("#anchors").val().split("\n"));
    $.get("/topics", {anchors: anchorArr}, update);
  });
});
*/
