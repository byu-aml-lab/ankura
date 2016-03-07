var app = angular.module('anchorApp', [])
    .controller('anchorController', function($scope, $timeout, $http) {
        var ctrl = this;
        //This holds all of the anchor objects.
        //  An anchor holds both anchor words for a single anchor and topic words that describe that anchor.
        ctrl.anchors = [];
        // This hold previous states, so we can undo/redo
        ctrl.anchorsHistory = [];
        // This tells us where we are in anchorsHistory
        ctrl.historyIndex = 0;
        // This sets the UI state back one place in anchorsHistory
        //   if there is a state to go back to
        ctrl.undo = function() {
            if (ctrl.historyIndex > 0) {
                ctrl.anchors = getAnchorsArray(ctrl.anchorsHistory[ctrl.historyIndex-1]["anchors"],
                                               ctrl.anchorsHistory[ctrl.historyIndex-1]["topics"]);
                ctrl.historyIndex -= 1;
                ctrl.startChanging();
            }
            else {
                $("#undoForm").popover({
                    placement:'top',
                    trigger:'manual',
                    html:true,
                    content:'Nothing to undo.'
                }).popover('show');
                $timeout(function() {
                    $("#undoForm").popover('hide');
                }, 1000);
            }
        }
        // This sets the UI state forward one place in anchorsHistory
        //   if there is a state to go forward to
        ctrl.redo = function() {
            if (ctrl.historyIndex+1 < ctrl.anchorsHistory.length) {
                ctrl.anchors = getAnchorsArray(ctrl.anchorsHistory[ctrl.historyIndex+1]["anchors"],
                                               ctrl.anchorsHistory[ctrl.historyIndex+1]["topics"]);
                ctrl.historyIndex += 1;
                ctrl.startChanging;
            }
            else {
                $("#redoForm").popover({
                    placement:'top',
                    trigger:'manual',
                    html:true,
                    content:'Nothing to redo.'
                }).popover('show');
                $timeout(function() {
                    $("#redoForm").popover('hide');
                }, 1000);
            }
        }
        // When finished is set to true, it brings us to the "thank you" page
        ctrl.finished = false;
        // This function sends the anchorsHistory array to the server
        //   and send the user to the "thank you" page
        ctrl.done = function() {
          var data = JSON.stringify(ctrl.anchorsHistory);
          $http.post("/finished", data).success(function(data, status) {
              ctrl.finished = true;
          });
        }
        // Vocab holds the vocabulary of valid words
        ctrl.vocab;
        $.get("/vocab", function(data) {
          ctrl.vocab = data.vocab;
        });
        // This function adds a blank anchor to the page
        ctrl.addAnchor = function() {
          var anchorObj = {"anchors":[], "topic":[]};
          ctrl.anchors.push(anchorObj);
          initAutocomplete(ctrl.vocab);
          ctrl.stopChanging();
        }
         //This function removes an anchor from the current list of anchors.
         //  it deletes a whole line (both anchor words and their topic words).
        ctrl.removeAnchor = function(index) {
          ctrl.anchors.splice(index, 1);
          ctrl.stopChanging();
        }
        //This function adds an anchor word when entered in via an input in the left column
        ctrl.addAnchorWord = function(textForm, newAnchor) {
            $scope.$broadcast("autofillfix:update"); //Needed to make autofill and Angular work well together
            var lowercaseAnchor = textForm.target.children[0].value.toLowerCase();
            //We are checking to see if the new anchor word is in the vocabulary.
            //  If it is, we add a new anchor and prompt to update topics.
            //  If it is not, we prompt to add a valid anchor.
            var inVocab = false;
            for (var i = 0; i < ctrl.vocab.length; i++) {
                if (ctrl.vocab[i] === lowercaseAnchor) inVocab = true;
             }
            if (inVocab) {
                newAnchor.push(lowercaseAnchor);
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
                textForm.target.children[0].value = "";
                ctrl.stopChanging();
            }
            else {
                angular.element(textForm.target).popover({
                    placement:'bottom',
                    trigger:'manual',
                    html:true,
                    content:'Invalid anchor word.'
                }).popover('show');
                $timeout(function() {
                    angular.element(textForm.target).popover('hide');
                }, 2000);
            }
        }

        //This function deletes an anchor word (when you click on the little 'x' in the bubble)
        ctrl.deleteWord = function(closeButton, array) {
            var toClose = closeButton.target.parentNode.id;
            $("#"+toClose).remove();
            console.log(array);
            var wordIndex = array.indexOf(closeButton.target.parentNode.textContent.replace(/✖/, "").replace(/\s/g, ''));
            if (wordIndex !== -1) {
                array.splice(wordIndex, 1);
            }
            ctrl.stopChanging();
        }
        //This function only gets the topics when we have no current anchors.
        ctrl.getTopics = function() {
            ctrl.loading = true;
            $.get("/topics", {example:''}, function(data) {
                //Ensure we can't redo something that's been written over
                ctrl.anchorsHistory.splice(ctrl.historyIndex, ctrl.anchorsHistory.length-ctrl.historyIndex-1);
                //Save the data
                console.log(data);
                ctrl.anchorsHistory.push(data);
                ctrl.anchors = getAnchorsArray(data["anchors"], data["topics"]);
                ctrl.getExampleDocuments(data['example']);
                ctrl.loading = false;
                ctrl.startChanging();
                $scope.$apply();
                initAutocomplete();
                $(".top-to-bottom").css("height", $(".anchors-and-topics").height());
            });
        }
        //We actually call the above function here, so we get the original topics
        ctrl.getTopics();
        //This function takes all anchors from the left column and gets their new topic words.
        //  It then updates the page to include the new topic words.
        ctrl.getNewTopics = function() {
            var currentAnchors = [];
            //The server throws an error if there are no anchors,
            //  so we want to get new anchors if needed.
            if ($(".anchorContainer").length !== 0) {
                $(".anchorContainer").each(function() {
                    //This parses out just the comma-separated anchors from all the html
                    var value = $(this).html().replace(/\s/g, '').replace(/<span[^>]*>/g, '').replace(/<\/span><\/span>/g, ',');
                    value = value.replace(/<!--[^>]*>/g, '').replace(/,$/, '').replace(/,$/, '').replace(/\u2716/g, '');
                    //This prevents errors on the server if there are '<' or '>' symbols in the anchors
                    value = value.replace(/\&lt;/g, '<').replace(/\&gt;/g, '>');
                    if (value === "") {
                        return true;
                    }
                    var tempArray = value.split(",");
                    currentAnchors.push(tempArray);
                });
                if (currentAnchors.length !== 0) {
                    var getParams = JSON.stringify(currentAnchors);
                    ctrl.loading = true;
                    $.get("/topics", {anchors: getParams, example:''}, function(data) {
                        var saveState = {anchors: currentAnchors,
                                   topics: data["topics"]};
                        //This gets rid of the possibility of redoing if another state was saved since the last undo. If nothing has been undone, this should do nothing.
                        ctrl.anchorsHistory.splice(ctrl.historyIndex+1, ctrl.anchorsHistory.length-ctrl.historyIndex-1);
                        //Increment historyIndex
                        ctrl.historyIndex += 1;
                        //Save the current state (anchors and topic words)
                        ctrl.anchorsHistory.push(saveState);
                        //Update the anchors in the UI
                        ctrl.anchors = getAnchorsArray(currentAnchors, data["topics"]);
                        ctrl.getExampleDocuments(data['example']);
                        ctrl.loading = false;
                        ctrl.startChanging();
                        $scope.$apply();
                        initAutocomplete();
                        // Sets the height of the document container
                        $(".top-to-bottom").css("height", $(".anchors-and-topics").height());
                    });
                }
                else {
                    ctrl.getTopics();
                }
            }
            //This gets new anchors if we need them.
            else {
                ctrl.getTopics();
            }
            initAutocomplete();
        }
        //This initializes autocompletion for entering new anchor words
        var initAutocomplete = function initAutocomplete() {
            $(".anchorInput" ).autocomplete({
                minLength: 3,
                source: ctrl.vocab,
                // This function is called whenever a list choice is selected
                select: function(event, ui) {
                  // This sets a listener to prevent the page from reloading
                  $(this).parents("form").on('submit', function() {
                    return false;
                  });
                  // This triggers the submit event, which turns the selected
                  //   word into a proper anchor word (with the border)
                  $(this).parents("form").submit();
                  // This prevents the value from being duplicated
                  return false;
                }
            });
        };
        // This sets the height of the document container on load
        $timeout(function() {
          $(".top-to-bottom").css("height", $(".anchors-and-topics").height());
        }, 50);
        // Holds all of the sample documents we were given
        ctrl.documents;
        // Holds a map from document to topics it includes
        ctrl.docToTopicList;
        // Holds a map from topic to documents that include it
        ctrl.topicToDocList;
        // Will get documents and docToTopicList from an endpoint, currently just test data
        ctrl.getDocumentsTest = function getDocumentsTest() {
          // We don't get data in this format anymore, we get a list of
          //   document/topic listing pairs
          var data = {documents: ['Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et lobortis risus. Aliquam risus ex, elementum non odio sit amet, euismod faucibus ligula. Fusce ut tortor diam. Integer finibus varius velit. Nullam ultricies, sem nec dictum fermentum, nunc lacus posuere nunc, non porttitor enim ipsum id nulla. Aliquam dictum cursus felis, id lobortis urna ornare at. Suspendisse laoreet, augue a commodo semper, tellus quam finibus leo, nec dapibus nibh est feugiat ligula. Proin non mauris dui.',
            'Cras sodales orci diam, a mattis lacus sagittis id. Nullam pellentesque urna congue turpis lobortis, sed consectetur enim euismod. Vivamus efficitur iaculis felis a tristique. Praesent suscipit porttitor orci in porttitor. Duis vel pellentesque ligula. Aenean cursus volutpat lectus et porta. Nulla tempor metus et interdum pellentesque. Integer dui tellus, iaculis eu lectus vestibulum, hendrerit porta arcu. Cras interdum odio tortor, quis finibus diam scelerisque quis.',
            'Sed convallis mollis metus, nec suscipit est porta nec. Mauris vitae efficitur odio, vel semper lacus. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Quisque rhoncus consectetur tortor eget aliquam. Duis non pretium tellus, vitae porttitor urna. Mauris et venenatis odio. Sed imperdiet gravida urna, ac vestibulum nunc condimentum ac. Aenean semper sollicitudin felis in venenatis. Etiam aliquam aliquam eros in ultricies. Sed ut eros ac augue molestie dignissim. Ut fringilla molestie risus id tristique. Suspendisse eget scelerisque nibh, ut volutpat neque. Donec rhoncus mauris nec orci facilisis luctus. Donec finibus efficitur fringilla.',
            'Aenean in orci nibh. Curabitur non enim et justo accumsan eleifend. Mauris sit amet dolor mattis dolor lobortis tincidunt vitae eu velit. In ut magna porttitor, laoreet purus et, sagittis ipsum. Cras fringilla libero eget lectus vehicula consequat. Sed hendrerit convallis blandit. Etiam posuere dapibus metus, eu iaculis nisl scelerisque ut. Donec posuere tellus felis, at feugiat magna facilisis eu. Nulla facilisi. Cras varius consequat est, quis luctus magna hendrerit sed.',
            'Sed vestibulum diam leo. Pellentesque luctus enim justo, quis fermentum lorem tristique nec. Proin sed ultrices arcu. Maecenas ut nibh tellus. Aenean mollis risus vitae mi ornare, eu venenatis eros vestibulum. Nunc lectus massa, accumsan in molestie vitae, cursus vitae justo. Mauris vel ex velit. Ut aliquam nisi nec ipsum condimentum gravida.'],
            topics: [[3, 6], [10, 12], [1, 8, 14], [1, 2, 3], [8, 10, 16]]};
          ctrl.documents = data['documents'];
          ctrl.docToTopicList = data['topics'];
          ctrl.topicToDocList = [];
          for (var i = 0; i < ctrl.docToTopicList.length; i++) {
            for (var j = 0; j < ctrl.docToTopicList[i].length; j++) {
              if (ctrl.topicToDocList[ctrl.docToTopicList[i][j]] === undefined) {
                ctrl.topicToDocList[ctrl.docToTopicList[i][j]] = [];
                ctrl.topicToDocList[ctrl.docToTopicList[i][j]].push(i);
              }
              else { ctrl.topicToDocList[ctrl.docToTopicList[i][j]].push(i); }
            }
          }
        }
        // Gets example documents to display on the right-hand side
        ctrl.getExampleDocuments = function getExampleDocuments(exampleDocs) {
          // Only get the first 5 docs for now, might change later
          ctrl.documents = exampleDocs.slice(0, 5);
          ctrl.topicToDocList = [];
          for (var i = 0; i < ctrl.documents.length; i++) {
            for (var j = 0; j < ctrl.documents[i]['topics'].length; j++) {
              console.log(ctrl.documents[i]['topics'][j]);
              if (ctrl.topicToDocList[ctrl.documents[i]['topics'][j]] === undefined) {
                ctrl.topicToDocList[ctrl.documents[i]['topics'][j]] = [];
                ctrl.topicToDocList[ctrl.documents[i]['topics'][j]].push(i);
              }
              else { ctrl.topicToDocList[ctrl.documents[i]['topics'][j]].push(i); }
            }
          }
        }
        ctrl.stopChanging = function stopChanging() {
          ctrl.noChangesYet = false;
          $('.document').css('background-color', '#FFFFFF');
          $('.anchor-and-topic').css('border', 'solid 2px #FFFFFF');
        }
        ctrl.startChanging = function startChanging() {
          ctrl.noChangesYet = true;
        }
        ctrl.addHighlightsDoc = function addHighlightsDoc(event, index) {
          if (ctrl.noChangesYet) {
            angular.element(event.target).css('background-color', '#FFFF55');
            var list = ctrl.documents[index]['topics'];
            for (var i = 0; i < list.length; i++) {
              $('#anchor-and-topic-'+list[i]).css('border', 'solid 2px #F0F055');
            }
          }
        }
        ctrl.removeHighlightsDoc = function removeHighlightsDoc(event, index) {
          if (ctrl.noChangesYet) {
            angular.element(event.target).css('background-color', '#FFFFFF');
            var list = ctrl.documents[index]['topics'];
            for (var i = 0; i < list.length; i++) {
              $('#anchor-and-topic-'+list[i]).css('border', 'solid 2px #FFFFFF');
            }
          }
        }
        ctrl.addHighlightsTopic = function addHighlightsTopic(event, index) {
          if (ctrl.noChangesYet) {
            var list = ctrl.topicToDocList[index];
            if (list !== undefined) {
              $('#anchor-and-topic-'+index).css('border', 'solid 2px #F0F055');
              for (var i = 0; i < list.length; i++) {
                $('#document-'+list[i]).css('background-color', '#FFFF55');
              }
            }
          }
        }
        ctrl.removeHighlightsTopic = function removeHighlightsTopic(event, index) {
          if (ctrl.noChangesYet) {
            var list = ctrl.topicToDocList[index];
            if (list !== undefined) {
              $('#anchor-and-topic-'+index).css('border', 'solid 2px #FFFFFF');
              for (var i = 0; i < list.length; i++) {
                $('#document-'+list[i]).css('background-color', '#FFFFFF');
              }
            }
          }
        }
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
        //Need to cover all the possible places in the main div it could be dropped
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
            $(ev.target).parent().parent().siblings(".anchorContainer")[0].appendChild(document.getElementById(data));
        }
        else if ($(ev.target).hasClass( "anchor" )) {
            $(ev.target).children(".anchorContainer")[0].appendChild(document.getElementById(data));
        }
        var $scope = angular.element('body').scope();
        $scope.$apply(function() {
          $scope.ctrl.stopChanging();
        });
    }
    //If a topic word, copy it
    else {
        var nodeCopy = document.getElementById(data).cloneNode(true);
        nodeCopy.id = data + "copy" + copyId++;
        var closeButton = addDeleteButton(nodeCopy.id + "close");
        nodeCopy.appendChild(closeButton);
        //Need to cover all the possible places in the main div it could be dropped
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
            $(ev.target).parent().parent().siblings(".anchorContainer")[0].appendChild(nodeCopy);
        }
        else if ($(ev.target).hasClass( "anchor" )) {
            $(ev.target).children(".anchorContainer")[0].appendChild(nodeCopy);
        }
        var $scope = angular.element('body').scope();
        $scope.$apply(function() {
          $scope.ctrl.stopChanging();
        });
    }
};

//used to delete words that are copies (because they can't access the function in the Angular scope)
var deleteWord = function(ev) {
    $("#"+ev.target.id).parent()[0].remove();
}

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
