
function update(data) {
  $("#topics pre").html(
    JSON.stringify(data["topics"])
      .replace(/\],/g, "\n")
      .replace(/["\[\]]/g, "")
  )
}

$(document).ready(function() {
  $.get("/topics", function(data) {
    $("#anchors").val(JSON.stringify(data["anchors"]).replace(/,/g, ",\n"));
    update(data);
  });
  $("#anchor-form").submit(function(e) {
    e.preventDefault();
    $.get("/topics", {anchors: $("#anchors").val()}, update);
  });
});
