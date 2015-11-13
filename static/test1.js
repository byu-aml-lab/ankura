angular.module('anchorApp', [])
  .controller('anchorController', function($scope) {
    var ctrl = this;
    ctrl.anchors = [];
    ctrl.addAnchor = function() {
      //TODO: Ensure new anchor is in the vocabulary
      var anchorObj = {"anchor":$scope.newAnchor,"topic":["Press Update Topics to get topics for this anchor"]};
      ctrl.anchors.push(anchorObj);
      $scope.newAnchor = '';
    }
    ctrl.removeAnchor = function(index) {
      ctrl.anchors.splice(index, 1);
    }
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
        console.log($(this).html().replace(/<span[^>]*>/g, '').replace(/<\/span>/g, ',').replace(/,$/, ''));
        var value = $(this).html().replace(/<span[^>]*>/g, '').replace(/<\/span>/g, ',').replace(/,$/, '');
        console.log(value);
        var tempArray = value.split(",");
        console.log(tempArray);
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
    var topic = topics[i];
    tempAnchors.push({"anchor":anchor, "topic":topic});
  }
  return tempAnchors;
};

var allowDrop = function(ev) {
  ev.preventDefault();
};

var drag = function(ev) {
  ev.dataTransfer.setData("text", ev.target.id);
};

//Holds id addition for when we copy nodes
var copyId = 0;

var drop = function(ev) {
  ev.preventDefault();
  var data = ev.dataTransfer.getData("text");
  console.log(data);
  console.log(JSON.stringify(data));
  var dataString = JSON.stringify(data);
  //If an anchor or a copy of a topic word, drop
  if (dataString.indexOf("anchor") !== -1 || dataString.indexOf("copy") !== -1) {
    ev.target.appendChild(document.getElementById(data));
  }
  //If a topic word, copy it
  else {
    var nodeCopy = document.getElementById(data).cloneNode(true);
    nodeCopy.id = data + "copy" + copyId++;
    ev.target.appendChild(nodeCopy);
  }
};

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
