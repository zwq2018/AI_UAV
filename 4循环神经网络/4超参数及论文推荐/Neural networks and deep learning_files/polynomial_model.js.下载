// Note that this script relies on globals defined in simple_data.js,
// such as data, width, height, and so on.  

var svg = d3.select("#polynomial_fit").append("svg") 
    .attr("width", width)
    .attr("height", height) 

// Polynomial model 

var line = d3.svg.line()
    .x(function(d) { return xScale(d[0]); }) 
    .y(function(d) { return yScale(d[1]); }); 

function p(x) {return 2.20539187e-01*Math.pow(x, 9)
	       -5.49142821*(Math.pow(x, 8)) 
	       +5.87844045e+01*(Math.pow(x, 7))
	       -3.53892824e+02*(Math.pow(x, 6)) 
	       +1.31549254e+03*(Math.pow(x, 5))
	       -3.11809836e+03*(Math.pow(x, 4)) 
	       +4.69080366e+03*(Math.pow(x, 3))
	       -4.29612493e+03*(Math.pow(x, 2)) 
	       +2.16228823e+03*x 
	       -4.50983951e+02;};

var polynomialFit = d3.range(0, 5, .01).map(function(d){ 
    return [d, p(d)]; }); 

svg.append("path") 
    .datum(polynomialFit) 
    .attr("class", "path") 
    .attr("d", line); 

svg.selectAll("circle") 
    .data(data) 
    .enter()
    .append("circle") 
    .attr("cx", function(d) {return xScale(d[0]);})
    .attr("cy", function(d) {return yScale(d[1]);}) 
    .attr("r", 5); 

// x axis 
svg.append("g") 
    .attr("class", "axis") 
    .attr("transform", "translate(0,"+(height-padding)+")") 
    .call(xAxis); 
svg.append("text")
    .attr("class", "axisLabel") 
    .attr("x", width/2) .attr("y", height - 3)
    .text("x"); 

// y axis 
svg.append("g") 
    .attr("class", "axis")
    .attr("transform", "translate("+padding+", 0)") 
    .call(yAxis);
svg.append("text") 
    .attr("class", "axisLabel") 
    .attr("x", 2)
    .attr("y", height/2) 
    .text("y");
