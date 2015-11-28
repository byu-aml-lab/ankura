angular.module('anchorApp', [])
  .controller('anchorController', function($scope) {
    var ctrl = this;
    ctrl.anchors = [];
    ctrl.vocab;
    $.get("/vocab", function(data) {
        ctrl.vocab = data.vocab;
    });    
    ctrl.addAnchor = function() {
      //TODO: Ensure new anchor is in the vocabulary
      $scope.$broadcast("autofillfix:update");
      var lowercaseAnchor = $scope.newAnchor.toLowerCase();
      var inVocab = false;
      for (var i = 0; i < ctrl.vocab.length; i++) {
          if (ctrl.vocab[i] === lowercaseAnchor) inVocab = true;
      }
      console.log(inVocab);
      if (inVocab) {
          var newAnchors = lowercaseAnchor.split(',');
          var anchorObj = {"anchors":newAnchors,"topic":["Press Update Topics to get topics for this anchor"]};
          ctrl.anchors.push(anchorObj);
          $scope.newAnchor = '';
      }
      else {
          $("#addAnchorInput").popover({
              placement:'top',
              trigger:'manual',
              html:true,
              content:'Please enter a valid vocabulary word'
          }).popover('show');
          //TODO: Set a timeout here
          $("#addAnchorInput").popover('hide');
      }
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
        var value = $(this).html().replace(/<span[^>]*>/g, '').replace(/<\/span><\/span>/g, ',');
        value = value.replace(/<!--[^>]*>/g, '').replace(/,$/, '').replace(/,$/, '').replace(/\s\u2716/g, '');
        if (value === "") {
            return true;
        }
        var tempArray = value.split(",");
        currentAnchors.push(tempArray);
      });
      var getParams = JSON.stringify(currentAnchors);
      $.get("/topics", {anchors: getParams}, function(data) {
        ctrl.anchors = getAnchorsArray(currentAnchors, data["topics"]);
        $scope.$apply();
      });
    }
    var initAutocomplete = function() {
        $.get("/vocab", function(data) {
            $( "#addAnchorInput" ).autocomplete({
                minLength: 3,
                source: data.vocab
            });
        });
    };
    initAutocomplete();
  }).directive("autofillfix", function() {
        //This is required because of some problem between Angular and autofill
        return {
            require: "ngModel",
            link: function(scope, element, attrs, ngModel) {
                scope.$on("autofillfix:update", function() {
                    ngModel.$setViewValue(element.val());
                });
            }
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

var deleteWord = function(closeButton) {
    console.log(closeButton);
    var toClose = closeButton.target.parentNode.id;
    console.log(toClose);
    $("#"+toClose).remove();
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
    var closeButton = addDeleteButton(nodeCopy.id + "close");
    nodeCopy.appendChild(closeButton);
    console.log(nodeCopy);
    if($(ev.target).hasClass( "droppable" )) {
      ev.target.appendChild(nodeCopy);
    }
    if($(ev.target).hasClass( "draggable" )) {
      $(ev.target).parent()[0].appendChild(nodeCopy);
    }
  }
};

var addDeleteButton = function(id) {
    var closeButton = document.createElement("span");
    closeButton.innerHTML = " &#10006"
    var closeClass = document.createAttribute("class");
    closeClass.value = "close";
    closeButton.setAttributeNode(closeClass);
    var closeId = document.createAttribute("id");
    closeId.value = id;
    closeButton.setAttributeNode(closeId);
    var closeClick = document.createAttribute("onclick");
    closeClick.value = "deleteWord(event)";
    closeButton.setAttributeNode(closeClick);
    return closeButton
};
