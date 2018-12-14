CanvasRenderingContext2D.prototype.clear = function() {
    // Clear the canvas corresponding to this context
    this.canvas.width = this.canvas.width;
}
    
CanvasRenderingContext2D.prototype.text = function(content, x, y, args) {
    // Note that the optional arguments are based on CSS property
    // names, not the canvas property names.  The exception is
    // "textBaseline", because the values for canvas are not modelled
    // on those for CSS.
    var args = parseArgs(args, {"font": defaultFont,
				"color": defaultText,
				"text-align": "left",
				"textBaseline": "alphabetic"});
    this.font = args.font;
    this.fillStyle = args.color;
    this.textAlign = args["text-align"];
    this.textBaseline = args.textBaseline;
    this.fillText(content, x, y);
}

CanvasRenderingContext2D.prototype.mathText = function(content, x, y, args) {
    // Renders content at x, y on the canvas.  Note that we can use ^
    // and _ to denote superscripts and subscripts, respectively.  So,
    // for example, f^1(x_2) renders as f with a 1 superscript, with
    // an argument of x with a 2 subscript.  args is as for the .text
    // method.  Note that only text-align: left and text-align: center
    // are currently supported.
    var args = parseArgs(args, {"size": 20,
				"littleSize": 14,
				"color": defaultMathText,
				"text-align": "left", 
			        "up": -8,
			        "down": 4});
    var dx = 0;
    var dy = 0;
    var fontMath = String(args.size)+"px MJX_Math"; // for alphabet
    var fontMain = String(args.size)+"px MJX_Main"; // for numbers
    var littleFontMath = String(args.littleSize)+"px MJX_Math";
    var littleFontMain = String(args.littleSize)+"px MJX_Main";
    this.fillStyle = args.color;
    this.textAlign = "left";
    var little = false;
    if (args["text-align"] === "center") {
	// adjust the starting x value
	var width = 0;
	content.split('').forEach(function(s) {
	    if (s === "_" || s === "^") {
		if (s === "^") {width += 2};
		little = true;
	    } else {
		if (!little && !isAlpha(s)) {this.font = fontMain};
		if (!little && isAlpha(s)) {this.font = fontMath};
		if (little && !isAlpha(s)) {this.font = littleFontMain};
		if (little && isAlpha(s)) {this.font = littleFontMath};
		width += this.measureText(s).width;
		little = false;
	    }
	}, this);
	x -= Math.round(width/2);
    }
    little = false;
    content.split('').forEach(function(s) {
	if (s === '^') {
	    dx += 2;
	    dy = args.up;
	    little = true;
	} else if (s === "_") {
	    dy = args.down;
	    little = true;
	} else {
	    if (!little && !isAlpha(s)) {this.font = fontMain};
	    if (!little && isAlpha(s)) {this.font = fontMath};
	    if (little && !isAlpha(s)) {this.font = littleFontMain};
	    if (little && isAlpha(s)) {this.font = littleFontMath};
	    this.fillText(s, x+dx, y+dy);
	    dx += this.measureText(s).width;
	    dy = 0;
	    little = false;
	}
    }, this);
}

CanvasRenderingContext2D.prototype.line = function(
    x1, y1, x2, y2, color, lineWidth) {
    this.beginPath();
    this.moveTo(x1, y1);
    this.lineTo(x2, y2);
    this.strokeStyle = selfOrDefault(color, "black");
    this.lineWidth = selfOrDefault(lineWidth, 1);
    this.stroke();
}

CanvasRenderingContext2D.prototype.arrow = function(
    x1, y1, x2, y2, color, lineWidth) {
    this.line(x1, y1, x2, y2, color, lineWidth);
    var delta = { // difference vector between the two points
	"x": x1-x2,
	"y": y1-y2
    }
    var norm = Math.sqrt(delta.x * delta.x + delta.y * delta.y);
    var n = { // normalized difference vector, pointing from pt 2 to pt 1
	"x": delta.x / norm,
	"y": delta.y / norm
    }
    var m = { // vector orthogonal to n
	"x": -n.y,
	"y": n.x
    }
    var arrow1 = { // part of the arrowhead
	"x": 8*n.x+4*m.x,
	"y": 8*n.y+4*m.y
    }
    var arrow2 = { // other part of the arrowhead
	"x": 8*n.x-4*m.x,
	"y": 8*n.y-4*m.y
    }
    this.line(x2, y2, x2+arrow1.x, y2+arrow1.y, color, lineWidth);
    this.line(x2, y2, x2+arrow2.x, y2+arrow2.y, color, lineWidth);
}

CanvasRenderingContext2D.prototype.filledRectangle = function(
    x1, y1, x2, y2, color, fillColor, lineWidth) {
    this.strokeStyle = selfOrDefault(color, "black");
    this.fillStyle = selfOrDefault(fillColor, "white");
    this.lineWidth = selfOrDefault(lineWidth, 1);
    this.fillRect(x1, y1, x2-x1, y2-y1);
    this.strokeRect(x1, y1, x2-x1, y2-y1);
}

CanvasRenderingContext2D.prototype.filledRoundedRectangle = function(
    x1, y1, x2, y2, r, color, fillColor, lineWidth) {
    this.strokeStyle = selfOrDefault(color, "black");
    this.fillStyle = selfOrDefault(fillColor, "white");
    this.lineWidth = selfOrDefault(lineWidth, 1);
    this.beginPath();
    this.moveTo(x1+r, y1);
    this.lineTo(x2-r, y1);
    this.quadraticCurveTo(x2, y1, x2, y1+r);
    this.lineTo(x2, y2-r);
    this.quadraticCurveTo(x2, y2, x2-r, y2);
    this.lineTo(x1+r, y2);
    this.quadraticCurveTo(x1, y2, x1, y2-r);
    this.lineTo(x1, y1+r);
    this.quadraticCurveTo(x1, y1, x1+r, y1);
    this.closePath();
    this.stroke();
    this.fill();
}

CanvasRenderingContext2D.prototype.circle = function(
    x, y, r, color, lineWidth, fillColor) { 
    // This is slower but produces smoother circles than the arc based method
    color = selfOrDefault(color, "black");
    lineWidth = selfOrDefault(lineWidth, 1);
    this.beginPath();
    var increment = 3/r;
    var x1 = x+r, y1 = y, x2, y2;
    this.moveTo(x1, y1);
    for (var theta = increment; theta < 2*Math.PI+increment; theta += increment) {
	x2 = x+r*Math.cos(theta);
	y2 = y+r*Math.sin(theta);
	this.lineTo(x2, y2);
	x1 = x2, y1 = y2;
    }
    this.strokeStyle = color;
    this.lineWidth = lineWidth;
    if (typeof fillColor !== "undefined") {
	this.fillStyle = fillColor;
    }
    this.stroke();
    this.fill();
}

CanvasRenderingContext2D.prototype.quickCircle = function(
    x, y, r, color, lineWidth) {
    this.beginPath();
    this.arc(x, y, r, 0, 2*Math.PI);
    this.strokeStyle = selfOrDefault(color, "black");
    this.lineWidth = selfOrDefault(lineWidth, 1);
    this.stroke();
}


function scale(in1, in2, out1, out2) {
    return function(x) {return (x-in1)*(out2-out1)/(in2-in1)+out1;}
}

CanvasRenderingContext2D.prototype.plot = function(
    data, xScale, yScale, color, lineWidth) {
    for (var j = 0; j < data.length-1; j++) {
	this.line(
	    xScale(data[j].x), yScale(data[j].y), 
	    xScale(data[j+1].x), yScale(data[j+1].y), 
	    color, lineWidth)
    }
}





