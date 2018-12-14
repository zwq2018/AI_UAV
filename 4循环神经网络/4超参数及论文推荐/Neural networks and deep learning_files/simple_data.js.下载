var data = [[ 0.7, 1.8], 
	    [ 1.3, 2.2], 
	    [ 1.9, 4.0], 
	    [ 2.6, 5.0], 
	    [ 2.9, 6.1], 
	    [ 3.6, 7.0], 
	    [ 3.8, 7.4], 
	    [ 3.95, 8.0], 
	    [ 4.4, 9.1], 
	    [ 4.9, 10.0] ]; 

var width = 520; 
var height = 360; 
var padding = 40; 

var xScale = d3.scale.linear() 
    .domain([0, 5]) 
    .range([padding, width-padding]);

var yScale = d3.scale.linear() 
    .domain([0, 10])
    .range([height-padding, padding]); 

var xAxis = d3.svg.axis()
    .scale(xScale) 
    .orient("bottom") 
    .ticks(5); 

var yAxis = d3.svg.axis()
    .scale(yScale) 
    .orient("left"); 

var svg = d3.select("#simple_model").append("svg") 
    .attr("width", width)
    .attr("height", height);

svg.selectAll("circle").data(data).enter()
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
    .attr("x", width/2) 
    .attr("y", height - 3)
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
