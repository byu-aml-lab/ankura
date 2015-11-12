angular.module('anchorApp', [])
  .controller('anchorController', function($scope) {
    var ctrl = this;
    ctrl.anchors = [];
    ctrl.getTopics = function() {
      $.get("/topics", function(data) {
        ctrl.anchors = getAnchorsArray(data["anchors"], data["topics"]);
        $scope.$apply();
      });
    }
    ctrl.getTopics();
    ctrl.getNewTopics = function() {
      var currentAnchors = [];
      $(".container .anchor").each(function() {
        var value = $(this).val().replace(/ /g, '');
        var tempArray = value.split(",");
        currentAnchors.push(tempArray);
      });
      var getParams = JSON.stringify(currentAnchors);
      $.get("/topics", {anchors: getParams}, function(data) {
        ctrl.anchors = getAnchorsArray(currentAnchors, data["topics"]);
        $scope.$apply();
      });
    }
  });

var getAnchorsArray = function(anchors, topics) {
  var tempAnchors = [];
  for (var i = 0; i < anchors.length; i++) {
    var anchor = JSON.stringify(anchors[i]).replace(/"/g, '').replace(/\[/g, '').replace(/\]/g, '');
    var topic = JSON.stringify(topics[i]).replace(/"/g, '').replace(/\[/g, '').replace(/\]/g, '');
    tempAnchors.push({"anchor":anchor, "topic":topic});
  }
  return tempAnchors;
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
