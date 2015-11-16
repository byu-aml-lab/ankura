angular.module('anchorApp', [])
  .controller('anchorController', function($scope) {
    var ctrl = this;
    ctrl.anchors = [];
    ctrl.addAnchor = function() {
      //TODO: Ensure new anchor is in the vocabulary
      var lowercaseAnchor = $scope.newAnchor.toLowerCase();
      var newAnchors = lowercaseAnchor.split(',');
      var anchorObj = {"anchors":newAnchors,"topic":["Press Update Topics to get topics for this anchor"]};
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
        var value = $(this).html().replace(/<span[^>]*>/g, '').replace(/<\/span>/g, ',').replace(/,$/, '');
        //Remove ng-repeat garbage left in the document, and another comma that somehow sneaks in
        value = value.replace(/<!--[^>]*>/g, '').replace(/,$/, '');
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
    anchor = anchors[i];
    var topic = topics[i];
    tempAnchors.push({"anchors":anchor, "topic":topic});
  }
  return tempAnchors;
};

var emptyTrash = function() {
    $("#trash").html("<h4 id=\"trashLabel\">Trash</h4><button onclick=\"emptyTrash()\" class=\"btn btn-danger btn-sm\" id=\"trashButton\">Empty Trash</button>");
}


//All functions below here enable dragging and dropping
//They could possibly be in another file and included?

var allowDrop = function(ev) {
  ev.preventDefault();
};

var drag = function(ev) {
  ev.dataTransfer.setData("text", ev.target.id);
};

//Holds next id for when we copy nodes
var copyId = 0;

var drop = function(ev) {
  ev.preventDefault();
  var data = ev.dataTransfer.getData("text");
  var dataString = JSON.stringify(data);
  //If an anchor or a copy of a topic word, drop
  if (dataString.indexOf("anchor") !== -1 || dataString.indexOf("copy") !== -1) {
    if($(ev.target).hasClass( "droppable" )) {
      ev.target.appendChild(document.getElementById(data));
    }
    if($(ev.target).hasClass( "draggable" )) {
      $(ev.target).parent()[0].appendChild(document.getElementById(data));
    }
  }
  //If a topic word, copy it
  else {
    var nodeCopy = document.getElementById(data).cloneNode(true);
    nodeCopy.id = data + "copy" + copyId++;
    if($(ev.target).hasClass( "droppable" )) {
      ev.target.appendChild(nodeCopy);
    }
    if($(ev.target).hasClass( "draggable" )) {
      $(ev.target).parent()[0].appendChild(nodeCopy);
    }
  }
};
