var defaultFont = "16px Arial";
var littleFont = "11px Arial";
var defaultLine = "#333"; 
var defaultText = "#333";
var defaultMathText = "#000";
var sliderColor = "#2A6EA6";

function isAlpha(s) {
    // Checks whether the string s contains alphabetical characters only
    return /^[a-zA-Z]+$/.test(s);
}

function selfOrDefault(x, deflt) {
    // Return x if x is not undefined, otherwise return the default
    return (typeof x !== "undefined") ? x : deflt;
}
function parseArgs(args, deflt) {
    // For each key in the associative array deflt, set args to that key,
    // if the key isn't in args.  Then return args.
    args = selfOrDefault(args, {});
    for (var index in deflt) {
	args[index] = selfOrDefault(args[index], deflt[index]);
    }
    return args;
}

$(function() {

    var zs = [2.5, -1, 3.2, 0.5]; // initial values for the z sliders
    function smG() {
	var Z = 0;
	for (var j=0; j <=3; j++) {
	    Z += Math.exp(zs[j]);
	} 
	var as = [0, 0, 0, 0];
	for (var j=0; j <=3; j++) {
	    var smGCanvas = $("#smG"+(j+1))[0];
	    var smGContext = smGCanvas.getContext("2d");
	    smGContext.clear();
            as[j] = Math.exp(zs[j])/Z;
	    smGContext.filledRectangle(5.5, 12.5, 205.5, 32.5, "#aaa", "white", 1);
	    smGContext.filledRectangle(
		6.5, 13.5, 6.5+as[j]*200, 31.5, sliderColor, sliderColor, 0);
            smGContext.line(5.5, 9.5, 5.5, 12.5, "#aaa");
	    smGContext.text("0", 2, 7, {"font": "10px Arial"});
            smGContext.line(205.5, 9.5, 205.5, 12.5, "#aaa");
	    smGContext.text("1", 202, 7, {"font": "10px Arial"});
	    $("#activation"+(j+1)).val(as[j].toFixed(3));
	};
    }

function createSlider(j) {
  $("#slider"+j).slider({
      range: "min", 
      min: -5, 
      max: 5, 
      value: zs[j-1], 
      step: 0.1,
      slide: function(event, ui) {
	  $("#amount"+j).val(ui.value);
	  zs[j-1] = ui.value;
	  smG();
      }
  });
}

    for (var j=1; j < 5; j++) {
	createSlider(j);
	$("#amount"+j).val($("#slider"+j).slider("value"));
	smG();
    }
});

