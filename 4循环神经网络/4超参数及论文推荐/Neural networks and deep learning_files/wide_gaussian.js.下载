var width = 600; 
var height = 120; 
var padding = 20; 

var sigma = Math.sqrt(501);
function gaussian(x) {
    return (1/(sigma*Math.sqrt(2*Math.PI)))*Math.exp(-x*x/(2*sigma*sigma));}

var xScale = d3.scale.linear() 
    .domain([-30, 30]) 
    .range([padding, width-padding]);

var yScale = d3.scale.linear() 
    .domain([0, 0.02])
    .range([height-padding, padding]); 

var line = d3.svg.line()
    .x(function(d) { return xScale(d[0]); }) 
    .y(function(d) { return yScale(d[1]); }); 

var xAxis = d3.svg.axis()
    .scale(xScale) 
    .orient("bottom") 
    .ticks(5);

var yAxis = d3.svg.axis()
    .scale(yScale) 
    .orient("left")
    .tickValues([0.02])
    .tickFormat(d3.format(".2"));

var svg = d3.select("#wide_gaussian").append("svg") 
    .attr("width", width)
    .attr("height", height);

var lineFit = d3.range(-30, 30, .1).map(function(d){ 
    return [d, gaussian(d)]; }); 

svg.append("path") 
    .datum(lineFit) 
    .attr("class", "path") 
    .attr("d", line); 

// x axis 
svg.append("g") 
    .attr("class", "axis") 
    .attr("transform", "translate(0,"+(height-padding)+")") 
    .call(xAxis); 

// y axis 
svg.append("g") 
    .attr("class", "axis")
    .attr("transform", "translate("+width/2+", 0)") 
    .call(yAxis);
