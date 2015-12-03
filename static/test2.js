angular.module('anchorApp', [])
    .controller('anchorController', function($scope, $timeout) {
        var ctrl = this;
        //This holds all of the anchor objects.
        //  An anchor holds both anchor words for a single anchor and topic words that describe that anchor.
        ctrl.anchors = [];
        ctrl.vocab;
        $.get("/vocab", function(data) {
            ctrl.vocab = data.vocab;
        });    
        //This function adds a new anchor (which is a new anchor word on a new line).
        //  It checks that the anchor word to be added is in the vocabulary.
        //  It also prompts the user to update the topics to get new words for the added anchor.
        ctrl.addAnchor = function() {
            $scope.$broadcast("autofillfix:update"); //Needed to make autofill and Angular work well together
            var lowercaseAnchor = $scope.newAnchor.toLowerCase();
            //We are checking to see if the new anchor word is in the vocabulary.
            //  If it is, we add a new anchor and prompt to update topics.
            //  If it is not, we prompt to add a valid anchor.
            var inVocab = false;
            for (var i = 0; i < ctrl.vocab.length; i++) {
                if (ctrl.vocab[i] === lowercaseAnchor) inVocab = true;
            }
            if (lowercaseAnchor === '') {
                var anchorObj = {"anchors":[], "topic":[]};
                ctrl.anchors.push(anchorObj);
            }
            else if (inVocab) {
                //The backend is expecting an array of anchor words, even if there's only one anchor word.
                var newAnchors = lowercaseAnchor.split(',');
                var anchorObj = {"anchors":newAnchors,"topic":[]};
                ctrl.anchors.push(anchorObj);
                $scope.newAnchor = '';
                //This timeout ensures that the added anchor is put in before the popover appears.
                //  If removed, the popover will appear too high above the "Update Topics" button.
                $timeout(function() {
                    $(".updateTopicsButtonClean").popover({
                        placement:'top',
                        trigger:'manual',
                        html:true,
                        content:'To see topic words for new anchors, press "Update Topics" here.'
                    }).popover('show')
                        .addClass("updateTopicsButtonDirty")
                        .removeClass("updateTopicsButtonClean");
                    //This timeout indicates how long the popover above will stay visible for.
                    $timeout(function() {
                        $(".updateTopicsButtonDirty").popover('hide')
                            .addClass("updateTopicsButtonClean")
                            .removeClass("updateTopicsButtonDirty");
                    }, 5000);
                }, 20);
            }
            else {
                //We want to ensure that the popover is not already visible, or it does wierd things when you
                //  hit the button rapidly and repeatedly.
                if ($(".addAnchorInputClean").length !== 0) {
                    $(".addAnchorInputClean").popover({
                        placement:'top',
                        trigger:'manual',
                        html:true,
                        content:'Please enter a valid vocabulary word'
                    }).popover('show').addClass("addAnchorInputDirty").removeClass("addAnchorInputClean");
                    //This timeout determines how long the message to enter a valid vocab word is visible for.
                    $timeout(function() {
                        $(".addAnchorInputDirty").popover('hide')
                            .addClass("addAnchorInputClean")
                            .removeClass("addAnchorInputDirty");
                    }, 2000);
                }
            }
        }
         //This function simply removes an anchor from the current list of anchors.
         //  In essence, it deletes a whole line (both anchor words and their topic words).
        ctrl.removeAnchor = function(index) {
            ctrl.anchors.splice(index, 1);
        }
        //This function only gets the topics when we have no current anchors.
        ctrl.getTopics = function() {
            $.get("/topics", function(data) {
                ctrl.anchors = getAnchorsArray(data["anchors"], data["topics"]);
                $scope.$apply();
            });
        }
        //We actually call the above function here, so we get the original topics
        ctrl.getTopics();
        //This function takes all anchors from the left column and gets their new topic words.
        //  It then repaints the page to include the new topic words.
        ctrl.getNewTopics = function() {
            var currentAnchors = [];
            //The server throws an error if there are no anchors, so we want to get new anchors if needed.
            if ($(".anchorContainer").length !== 0) {    
                $(".anchorContainer").each(function() {
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
            //This gets new anchors if we need them.
            else {
                ctrl.getTopics();
            }
        }           
        //This initializes autocompletion for entering new anchor words
        var initAutocomplete = function() {
            $.get("/vocab", function(data) {
                $(".addAnchorInputClean" ).autocomplete({
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

//This function returns an array of anchor objects from arrays of anchors and topics.
//Anchor objects hold both anchor words and topic words related to the anchor words.
var getAnchorsArray = function(anchors, topics) {
    var tempAnchors = [];
    for (var i = 0; i < anchors.length; i++) {
        anchor = anchors[i];
        var topic = topics[i];
        tempAnchors.push({"anchors":anchor, "topic":topic});
    }
    return tempAnchors;
};

//This function deleted an anchor word (when you click on the little 'x' in the bubble)
var deleteWord = function(closeButton) {
    var toClose = closeButton.target.parentNode.id;
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
        else if($(ev.target).hasClass( "draggable" )) {
            $(ev.target).parent()[0].appendChild(document.getElementById(data));
        }
        else if($(ev.target).hasClass( "anchorInputContainer" )) {
            $(ev.target).siblings(".anchorContainer")[0].appendChild(document.getElementById(data));
        }
        else if ($(ev.target).hasClass( "anchorInput" )) {
            $(ev.target).parent().siblings(".anchorContainer")[0].appendChild(document.getElementById(data));
        }
        else if ($(ev.target).hasClass( "anchor" )) {
            $(ev.target).children(".anchorContainer")[0].appendChild(document.getElementById(data));
        }
    }
    //If a topic word, copy it
    else {
        var nodeCopy = document.getElementById(data).cloneNode(true);
        nodeCopy.id = data + "copy" + copyId++;
        var closeButton = addDeleteButton(nodeCopy.id + "close");
        nodeCopy.appendChild(closeButton);
        console.log(ev.target);
        console.log($(ev.target));
        if($(ev.target).hasClass( "droppable" )) {
            ev.target.appendChild(nodeCopy);
        }
        else if($(ev.target).hasClass( "draggable" )) {
            $(ev.target).parent()[0].appendChild(nodeCopy);
        }
        else if($(ev.target).hasClass( "anchorInputContainer" )) {
            $(ev.target).siblings(".anchorContainer")[0].appendChild(nodeCopy);
        }
        else if ($(ev.target).hasClass( "anchorInput" )) {
            $(ev.target).parent().siblings(".anchorContainer")[0].appendChild(nodeCopy);
        }
        else if ($(ev.target).hasClass( "anchor" )) {
            $(ev.target).children(".anchorContainer")[0].appendChild(nodeCopy);
        }
    }
};

//Adds a delete button (little 'x' on the right side) of an anchor word
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
