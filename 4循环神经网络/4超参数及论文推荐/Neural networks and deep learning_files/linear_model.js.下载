// Note that this script relies on globals defined in simple_data.js,
// such as data, width, height, and so on.  

var svg = d3.select("#linear_fit").append("svg") 
    .attr("width", width)
    .attr("height", height) 

// Straight line model 

var linearModel = [[0,0], [5, 10]]; 

svg.append("path").datum(linearModel)
    .attr("class", "path") .attr("d", line);


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
