// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function deconv1d(){


  // Defaults
  const max_L = 2;
  const default_strides = [0,2,2];
  const default_sizes = [6,3,3];
  const max_sizes = [8,6];
  const display_layers = [0,1];

  var max_shown = 20;


  //==============================================================================
  //==============================================================================


  // SVG
  var fig = d3.select("#deconv1d");
  var is_wide = fig.node().getBoundingClientRect().width > 600;
  var svg = fig.append("svg");
  svg.style({width: '100%', "max-height": '130px'});
  svg.attr("version", "1.1")
  svg.attr("viewBox", "4 3.8 24 4")
  svg.attr("preserveAspectRatio", "xMidYMid meet")


  //==============================================================================
  //==============================================================================


  function iterate_tree(ns){
    if (ns.length == 0) return [[]];
    var rets = [];
    var subs = iterate_tree(ns.slice(1));
    for (var i = 0; i < ns[0]; i++)
    for (var sub of subs){
      rets.push([i].concat(sub));
    }
    return rets;
  }

  var line = d3.svg.line().x(d => d.x).y(d => d.y);

  //function show_level_simple()

  function draw_vis(sizes, strides){

    function iterate_tree(ns){
      if (ns.length == 0) return [[]];
      var rets = [];
      var subs = iterate_tree(ns.slice(1));
      for (var i = 0; i < ns[0]; i++)
      for (var sub of subs){
        rets.push([i].concat(sub));
      }
      return rets;
    }
    function normal_prep(t){
      var t2 = t.map((ti, i) => Math.min(ti, sizes[i]-1));
      return {t: t, t2: t2, collapsed: _.any(_.zip(t,t2).map(([t1i, t2i]) => t1i > t2i)) };
    }
    function tree_to_ind(t){
      var n = 0;
      for (var i in t){
        n = t[i] + strides[i]*n;
      }
      return n;
    }

    var max_widths = _.range(max_L).map(l => tree_to_ind(sizes.map(s => s-1).slice(0,l+1)) );

    var offsets = _.range(max_L).map(l => (30 - max_widths[l])/2);
    var x_spacing = {};
    var x_spacing_spread = {};
    for (var l = 0; l < max_L; l++){
      x_spacing[l] = _.range(500).map(n => offsets[l] + n);
      if (l < max_L - 1)
        x_spacing_spread[l] = _.range(500).map(n => offsets[l+1] + (sizes[l+1]-1)/2+strides[l+1]*n);
    }

    for (var l = 0; l < max_L; l++){
      if (display_layers.indexOf(l) == -1) continue;

      var data = iterate_tree(max_sizes.slice(0,l+1)).map(normal_prep);

      var visible = sizes.slice(1,l+1).reduce((a,b) => a*b, 1);
      var LF = Math.min(0.9, 1.5/visible);


      var g = svg.selectAll("#l_stride"+l).data([{}]);
      g.enter().append("g").attr("id", "l_stride"+l);
      var boxes = g.selectAll("path").data(data);
      boxes.enter().append("path")
        .attr("id", d => "box"+d.t)
        .attr("fill", "#000")
        .style('stroke', 'white')
        .style('stroke-width', 0.02)
        .style("opacity", 0.8*LF)
      boxes.transition().ease("out")
        .style("opacity", 0.8*LF)//d => d.collapsed? 0.2*LF : 0.8*LF)
        .attr('d', function stride_d(d) {
          var t = d.t;
          var dC = 1.5;
          var dX = d.collapsed? 0 : 1;
          if (l == max_L-1){
            var x = x_spacing[l][tree_to_ind(d.t2)+ (d.collapsed? 1 : 0)];
            var y1 = 7*l, y2 = 7*l + 0.5;
            return line([{x: x, y: y1}, {x: x, y: y2}, {x: x+dX, y: y2}, {x: x+dX, y: y1}]);
          }
          var x1 = x_spacing[l][tree_to_ind(d.t2)+ (d.collapsed? 1 : 0)];
          var x2 = x_spacing_spread[l][tree_to_ind(d.t2)] + (d.collapsed? 1 : 0);

          var y0 = 7*l, y1 = 7*l+0.5, y2 = 7*l+4, y3 = 7*l+4 + 0.5;
          if (l==0){
            return line([{x: x2, y: y2}, {x: x2, y: y3}, {x: x2+dX, y: y3}, {x: x2+dX, y: y2}, {x: x2, y: y2}]);
          }
          var p = d3.path();
          p.moveTo(x1, y0);
          p.lineTo(x1, y1);
          p.bezierCurveTo(x1, y1+dC, x2, y2-dC, x2, y2);
          p.lineTo(x2, y2);
          p.lineTo(x2, y3);
          p.lineTo(x2+dX, y3);
          p.lineTo(x2+dX, y2);
          p.bezierCurveTo(x2+dX, y2-dC, x1+dX, y1+dC, x1+dX, y1);
          p.lineTo(x1+dX, y0);
          //p.lineTo(X(x1+1), Y(y1));
          return p.toString();
          //line([{x: x2, y: y2}, {x: x2, y: y3}, {x: x2+1, y: y3}, {x: x2+1, y: y2}, {x: x1+1, y: y1}, {x: x1, y: y1}, {x: x2, y: y2}]);
        });

        var g = svg.selectAll("#l_outline"+l).data([{}]);
        g.enter().append("g").attr("id", "l_outline"+l);
        var boxes = g.selectAll("path").data(_.range(100));
        boxes.enter().append("path")
          .attr("id", n => "outline"+l+","+n)
          .attr("fill", "none")
          .style('stroke', 'black')
          .style('stroke-width', 0.02);
        boxes.transition().ease("out")
          .attr('d', function outline_d(n) {
            if (l == max_L-1){
              var width = max_widths[l];
              var collapsed = n > width;
              var dX = collapsed? 0 : 1;
              var n2 = Math.min(n, width)
              var x = x_spacing[l][n2+ (collapsed? 1 : 0)];
              var y1 = 7*l, y2 = 7*l + 0.5;
              return line([{x: x, y: y1}, {x: x, y: y2}, {x: x+dX, y: y2}, {x: x+dX, y: y1}, {x: x, y: y1}]);
            } else {
              var width = max_widths[l+1] - sizes[l+1] + 1;
              offset = (sizes[l+1]-1)/2;
              var collapsed = n > width;
              var dX = collapsed? 0 : 1;
              var n2 = Math.min(n, width);
              var x = x_spacing[l+1][n2+ (collapsed? 1 : 0)] + (sizes[l+1]-1)/2.0;
              var y2 = 7*l+4, y3 = 7*l+4 + 0.5;
              //if (l==0) console.log(x, sizes[l+1], x_spacing[l+1], n2+ (sizes[l+1]-1)/2.0+ (collapsed? 1 : 0));
              return line([{x: x, y: y2}, {x: x, y: y3}, {x: x+dX, y: y3}, {x: x+dX, y: y2}, {x: x, y: y2}]);
            }

            });

        //conv
        if (l < max_L-1){
          var g = svg.selectAll("#l_conv"+l).data([{}]);
          g.enter().append("g").attr("id", "l_conv"+l);
          var boxes = g.selectAll("path").data(data);
          boxes.enter().append("path")
            .attr("id", d => "conv"+d.t)
            .attr("fill", "#008")
            .style("opacity", d => 0.9*LF/sizes[d.t.length]);
          boxes.transition().ease("out").attr('d', function conv_d(d) {
            var n = tree_to_ind(d.t2);
            var x1 = x_spacing_spread[l][tree_to_ind(d.t2)];
            var x2 = x_spacing[l+1][tree_to_ind(d.t2.concat([0]))];
            var y1 = 7*l+4 + 0.5, y2 = 7*(l+1);
            if (d.collapsed){
              return line([{x: x1+1, y: y1}, {x: x1+1, y: y1}, {x: x2+sizes[l+1], y: y2}, {x: x2+sizes[l+1], y: y2}, {x: x1+1, y: y1},]);
            } else {
              return line([{x: x1, y: y1}, {x: x1+1, y: y1}, {x: x2+sizes[l+1], y: y2}, {x: x2, y: y2}, {x: x1, y: y1},]);
            }
          })
          .style("opacity", d => 0.9*LF/sizes[d.t.length]);

        }
    }

  }

  if (is_wide){
    fig.append("div")
      .style("position", "absolute")
      .style("top", "0px")
      .style("right", "0px")
      .style("width", "calc((100% - 648px)/2)")
      .style("height", "100%")
      .style("background", "linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,1), rgba(255,255,255,1), rgba(255,255,255,1), rgba(255,255,255,1))")

      fig.append("div")
        .style("position", "absolute")
        .style("top", "0px")
        .style("left", "0px")
        .style("width", "calc((100% - 648px)/4)")
        .style("height", "100%")
        .style("background", "linear-gradient(to left, rgba(255,255,255,0), rgba(255,255,255,1))")
    }

  var stride_ranges = [];
  var size_ranges = [];
  var stride_spans = [];
  var size_spans = [];

  //fig.append('br');
  for (var l = 0; l < max_L; l++){

    if (is_wide){
      var div = fig.append("div");
      if (l==0) div.style("display", "none")
      stride_spans.push(div.append('span')
        .style("position", "absolute")
        .style("top", (30 + 200*(l-1))  + "px")
        .style("right", "200px")
        .style("right", "calc((100% - 648px)/2 - 200px)")
      );
      size_spans.push(div.append('span')
        .style("position", "absolute")
        .style("top", (80 + 200*(l-1))  + "px")
        .style("right", "200px")
        .style("right", "calc((100% - 648px)/2 - 200px)")
      );

      stride_ranges.push(div.append('input').attr('type', 'range')
        .attr('min', 1).attr('max', max_sizes[l])
        .style("position", "absolute")
        .style("top", (20 + 200*(l-1)) + "px")
        .style("right", "200px")
        .style("right", "calc((100% - 648px)/2 - 200px)")
      );
      size_ranges.push(div.append('input')
        .attr('type', 'range').attr('min', 1).attr('max', max_sizes[l])
        .style("position", "absolute")
        .style("top", (70 + 200*(l-1))  + "px")
        .style("right", "200px")
        .style("right", "calc((100% - 648px)/2 - 200px)")
      );
    } else {
      var div = fig.append("div").style({float: "left", "margin": 0, "margin-right" : "20px"});
      if (l==0) div.style("display", "none")
      stride_ranges.push(div.append('input').attr('type', 'range')
        .attr('min', 1).attr('max', max_sizes[l])
      );
      div.append("br");
      stride_spans.push(div.append('span').style("margin-top", -5));
      var div = fig.append("div").style({float: "left", "margin-right" : "20px"});
      if (l==0) div.style("display", "none")
      size_ranges.push(div.append('input')
        .attr('type', 'range').attr('min', 1).attr('max', max_sizes[l])
      );
      div.append("br");
      size_spans.push(div.append('span'));
    }
  }
  if (!is_wide) {
    fig.append("br").style("clear", "left")
  }

  function update_vis(){
    var strides = [], sizes = [];
    for (var l = 0; l < max_L; l++){
      var stride = parseInt(stride_ranges[l].node().value);
      var size = parseInt(size_ranges[l].node().value);
      strides.push(stride);
      sizes.push(size);
      stride_spans[l].text("stride = " + stride);
      size_spans[l].text("size = " + size);
    }
    draw_vis(sizes, strides);
  }

  var strides = [0,2];
  var sizes = [8,3];
  for (var l = 0; l < max_L; l++){
    stride_ranges[l].node().value = strides[l];
    stride_ranges[l].on("input", update_vis);
    size_ranges[l].node().value = sizes[l];
    size_ranges[l].on("input", update_vis);
  }
  update_vis();

}


//===========================================================================
//===========================================================================
//===========================================================================

function deconv2d(){

  var fig = d3.select("#deconv2d");

  var resetIntervalId;

  var X = 13, Y = 9;
  var size = 3, stride = 2;

  var svg = fig.append("svg");
  //svg.style({width: '1000px', height: '350px'});
  svg.style({width: '100%', "max-height": '190px'});
  svg.attr("version", "1.1")
  svg.attr("viewBox", "0 0 1000 290")
  svg.attr("preserveAspectRatio", "xMidYMid meet")

  var C = d3.scale.linear()
      .domain([0, 4])
      .range(["white", "black"]);

  var proj = (p) => ({x: 20 + 68*(0.9*p.x + 0.3*p.y), y: 125 - 68*(p.z-0.25*p.y)});
  var proj_line = d3.svg.line().x(d => proj(d).x).y(d => proj(d).y);
  var draw_line = (path) => {
    svg.append('path')
      .attr('d', proj_line(path))
      .style('fill', 'none')
      .style('stroke', 'black')
      .style('stroke-width', 1);
  }

  var counts = _.range(30).map(x => _.range(30).map(y => 0));
  var rect_data = [];
   _.range(X).forEach(x => _.range(Y).forEach(y => {
     rect_data.push({x: x, y: y});
   }));

  var rects = svg.append('g').selectAll('path').data(rect_data);
  rects.enter().append('path')
    .style('fill', 'white')
    .style('stroke', 'black')
    .style('stroke-width', 1);
  rects.attr('d', d => proj_line([{x: d.x+0, y: d.y+0, z: 0}, {x: d.x+0, y: d.y+1, z: 0}, {x: d.x+1, y: d.y+1, z: 0},
                                    {x: d.x+1, y: d.y+0, z: 0}, {x: d.x+0, y: d.y+0, z: 0}]))

  var floater_g = svg.append('g').style("opacity", 0);

  var step_n = 0;
  var reset_pending = false;
  function reset(){
    step_n = 0;
    counts = counts.map(l => l.map(n => 0));
    rects.style('fill', d => C(counts[d.x][d.y]));
    reset_pending = false;
  }
  function step(){
    var box = fig.node().getBoundingClientRect();
    var height = window.innerHeight||document.documentElement.clientHeight;
    if (!(box.top + 0.3*box.height >= 0 && box.bottom - 0.2*box.height <= height)) return;
    if (document.visibilityState === "hidden") return;
    //if (svg_style.display === 'none') return;
    var nX = Math.ceil((X-size+1)/stride), nY = Math.ceil((Y-size+1)/stride);
    //console.log(nX, nY, step_n);
    if (step_n == nX*nY) {
      floater_g.style("opacity", 0);
      if (!reset_pending) {
        setTimeout(reset, 2000);
        reset_pending = true;
      }
      return;
    }
    var x = stride*(step_n % nX), y = stride*Math.floor(step_n/nX);
    x = ((y/stride) % 2 == 0)? x : stride*(nX-1) - x;
    step_n += 1;
    _.range(size).forEach(dX => _.range(size).forEach(dY => {
      counts[dX+x][dY+y] += 1;
    }));

    rects.transition().delay(200).duration(100).style('fill', d => C(counts[d.x][d.y]));
    var offset = (size-1)/2;

    var z_top = 2;
    var paths = [];
    paths.push([{x: x+0, y: y, z: 0}, {x: x+size, y: y, z: 0}, {x: x+size-offset, y: y+offset, z: z_top}, {x: x+offset, y: y+offset, z: z_top}, {x: x+0, y: y, z: 0}]);
    for (s of [1,0]) {
      paths.push([{x: x+s*size, y: y+0, z: 0}, {x: x+s*size, y: y+size, z: 0}, {x: x+offset+s, y: y+size-offset, z: z_top}, {x: x+offset+s, y: y+offset, z: z_top}, {x: x+s*size, y: y+0, z: 0}]);
    }
    for (s of [1]) {
      paths.push([{x: x+0, y: y+s*size, z: 0}, {x: x+size, y: y+s*size, z: 0}, {x: x+size-offset, y: y+offset+s, z: z_top}, {x: x+offset, y: y+offset+s, z: z_top}, {x: x+0, y: y+s*size, z: 0}]);
    }
    var temp = floater_g.selectAll('path').data(paths);
    //console.log(temp);
    temp.enter().append('path')
      .style('fill', '#99E')
      .style('stroke', 'black')
      .style('stroke-width', 1)
      .style("opacity", 0.7);
    temp.transition().duration(200).attr('d', d => proj_line(d));
    floater_g.transition().delay(200).duration(0).style("opacity", 0.5);
    floater_g
      .selectAll('g').data([{}]).enter().append('g')
      .selectAll('path').data([{}]).enter().append('path')
        .style('fill', '#777')
        .style('stroke', 'black')
        .style('stroke-width', 1)
        .style("opacity", 1);
    floater_g
      .select('g').select('path')
      .transition().duration(200)
      .attr('d', d => proj_line([{x: x+offset, y: y+offset, z: z_top}, {x: x+offset+1, y: y+offset, z: z_top}, {x: x+offset+1, y: y+offset+1, z: z_top}, {x: x+offset, y: y+offset+1, z: z_top}, {x: x+offset, y: y+offset, z: z_top}]));


    }

    setInterval(() => step(), 700);

}



//===========================================================================
//===========================================================================
//===========================================================================





function deconv1d_multi(){


  // Defaults
  const max_L = 3;
  const default_strides = [0,2,2];
  const default_sizes = [6,3,3];
  const max_sizes = [5,5,5];
  const display_layers = [0,1,2];

  var max_shown = 20;


  //==============================================================================
  //==============================================================================


  // SVG
  var fig = d3.select("#deconv1d_multi");
  var is_wide = fig.node().getBoundingClientRect().width > 600;
  var svg = fig.append("svg");
  svg.style({width: '100%', "max-height": '360px'});
  svg.attr("version", "1.1")
  svg.attr("viewBox", "0 3.8 35 11")
  svg.attr("preserveAspectRatio", "xMidYMid meet")


  //==============================================================================
  //==============================================================================


  function iterate_tree(ns){
    if (ns.length == 0) return [[]];
    var rets = [];
    var subs = iterate_tree(ns.slice(1));
    for (var i = 0; i < ns[0]; i++)
    for (var sub of subs){
      rets.push([i].concat(sub));
    }
    return rets;
  }

  var line = d3.svg.line().x(d => d.x).y(d => d.y);

  //function show_level_simple()

  function draw_vis(sizes, strides){

    function iterate_tree(ns){
      if (ns.length == 0) return [[]];
      var rets = [];
      var subs = iterate_tree(ns.slice(1));
      for (var i = 0; i < ns[0]; i++)
      for (var sub of subs){
        rets.push([i].concat(sub));
      }
      return rets;
    }
    function normal_prep(t){
      var t2 = t.map((ti, i) => Math.min(ti, sizes[i]-1));
      return {t: t, t2: t2, collapsed: _.any(_.zip(t,t2).map(([t1i, t2i]) => t1i > t2i)) };
    }
    function tree_to_ind(t){
      var n = 0;
      for (var i in t){
        n = t[i] + strides[i]*n;
      }
      return n;
    }

    var max_widths = _.range(max_L).map(l => tree_to_ind(sizes.map(s => s-1).slice(0,l+1)) );

    var base_delta = 26 - max_widths[max_L-1];
    if (is_wide) {
      var base_offset = 2+Math.min(base_delta/2, base_delta);
    } else {
      var base_offset = base_delta/2;
    }
    var offsets = _.range(max_L).map(l => base_offset + (max_widths[max_L-1] - max_widths[l])/2);
    var x_spacing = {};
    var x_spacing_spread = {};
    for (var l = 0; l < max_L; l++){
      x_spacing[l] = _.range(500).map(n => offsets[l] + n);
      if (l < max_L - 1)
        x_spacing_spread[l] = _.range(500).map(n => offsets[l+1] + (sizes[l+1]-1)/2+strides[l+1]*n);
    }

    for (var l = 0; l < max_L; l++){
      if (display_layers.indexOf(l) == -1) continue;

      var data = iterate_tree(max_sizes.slice(0,l+1)).map(normal_prep);

     //var visible = _.zip(sizes,strides).slice(1,l+1).map(([s,d]) => Math.ceil(s/d)).reduce((a,b) => a*b, 0.8);
     /*var visible = sizes.slice(1,l).reduce((a,b) => a*b, 1);
     function vis2opacity(n){
       return 1.5/n//1 - Math.pow(0.1/n, 1/n);
     }
     var LF = Math.min(0.9, vis2opacity(visible));*/
     var visible = sizes.slice(1,l+1).reduce((a,b) => a*b, 1);
     var LF = Math.min(0.9, 1.5/visible);


      var g = svg.selectAll("#l_stride"+l).data([{}]);
      g.enter().append("g").attr("id", "l_stride"+l);
      var boxes = g.selectAll("path").data(data);
      boxes.enter().append("path")
        .attr("id", d => "box"+d.t)
        .attr("fill", "#000")
        .style('stroke', 'white')
        .style('stroke-width', 0.02)
        .style("opacity", 0.8*LF)
      boxes.transition().ease("out")
        .style("opacity", 0.8*LF)//d => d.collapsed? 0.2*LF : 0.8*LF)
        .attr('d', function stride_d(d) {
          var t = d.t;
          var dC = 1.5;
          var dX = d.collapsed? 0 : 1;
          if (l == max_L-1){
            var x = x_spacing[l][tree_to_ind(d.t2)+ (d.collapsed? 1 : 0)];
            var y1 = 7*l, y2 = 7*l + 0.5;
            return line([{x: x, y: y1}, {x: x, y: y2}, {x: x+dX, y: y2}, {x: x+dX, y: y1}]);
          }
          var x1 = x_spacing[l][tree_to_ind(d.t2)+ (d.collapsed? 1 : 0)];
          var x2 = x_spacing_spread[l][tree_to_ind(d.t2)] + (d.collapsed? 1 : 0);

          var y0 = 7*l, y1 = 7*l+0.5, y2 = 7*l+4, y3 = 7*l+4 + 0.5;
          if (l==0){
            return line([{x: x2, y: y2}, {x: x2, y: y3}, {x: x2+dX, y: y3}, {x: x2+dX, y: y2}, {x: x2, y: y2}]);
          }
          var p = d3.path();
          p.moveTo(x1, y0);
          p.lineTo(x1, y1);
          p.bezierCurveTo(x1, y1+dC, x2, y2-dC, x2, y2);
          p.lineTo(x2, y2);
          p.lineTo(x2, y3);
          p.lineTo(x2+dX, y3);
          p.lineTo(x2+dX, y2);
          p.bezierCurveTo(x2+dX, y2-dC, x1+dX, y1+dC, x1+dX, y1);
          p.lineTo(x1+dX, y0);
          //p.lineTo(X(x1+1), Y(y1));
          return p.toString();
          //line([{x: x2, y: y2}, {x: x2, y: y3}, {x: x2+1, y: y3}, {x: x2+1, y: y2}, {x: x1+1, y: y1}, {x: x1, y: y1}, {x: x2, y: y2}]);
        });

        var g = svg.selectAll("#l_outline"+l).data([{}]);
        g.enter().append("g").attr("id", "l_outline"+l);
        var boxes = g.selectAll("path").data(_.range(100));
        boxes.enter().append("path")
          .attr("id", n => "outline"+l+","+n)
          .attr("fill", "none")
          .style('stroke', 'black')
          .style('stroke-width', 0.02);
        boxes.transition().ease("out")
          .attr('d', function outline_d(n) {
            if (l == max_L-1){
              var width = max_widths[l];
              var collapsed = n > width;
              var dX = collapsed? 0 : 1;
              var n2 = Math.min(n, width)
              var x = x_spacing[l][n2+ (collapsed? 1 : 0)];
              var y1 = 7*l, y2 = 7*l + 0.5;
              return line([{x: x, y: y1}, {x: x, y: y2}, {x: x+dX, y: y2}, {x: x+dX, y: y1}, {x: x, y: y1}]);
            } else {
              var width = max_widths[l+1] - sizes[l+1] + 1;
              offset = (sizes[l+1]-1)/2;
              var collapsed = n > width;
              var dX = collapsed? 0 : 1;
              var n2 = Math.min(n, width);
              var x = x_spacing[l+1][n2+ (collapsed? 1 : 0)] + (sizes[l+1]-1)/2.0;
              var y2 = 7*l+4, y3 = 7*l+4 + 0.5;
              //if (l==0) console.log(x, sizes[l+1], x_spacing[l+1], n2+ (sizes[l+1]-1)/2.0+ (collapsed? 1 : 0));
              return line([{x: x, y: y2}, {x: x, y: y3}, {x: x+dX, y: y3}, {x: x+dX, y: y2}, {x: x, y: y2}]);
            }

            });

        //conv
        if (l < max_L-1){
          var g = svg.selectAll("#l_conv"+l).data([{}]);
          g.enter().append("g").attr("id", "l_conv"+l);
          var boxes = g.selectAll("path").data(data);
          boxes.enter().append("path")
            .attr("id", d => "conv"+d.t)
            .attr("fill", "#008")
            .style("opacity", d => 0.9*LF/sizes[d.t.length]);
          boxes.transition().ease("out").attr('d', function conv_d(d) {
            var n = tree_to_ind(d.t2);
            var x1 = x_spacing_spread[l][tree_to_ind(d.t2)];
            var x2 = x_spacing[l+1][tree_to_ind(d.t2.concat([0]))];
            var y1 = 7*l+4 + 0.5, y2 = 7*(l+1);
            if (d.collapsed){
              return line([{x: x1+1, y: y1}, {x: x1+1, y: y1}, {x: x2+sizes[l+1], y: y2}, {x: x2+sizes[l+1], y: y2}, {x: x1+1, y: y1},]);
            } else {
              return line([{x: x1, y: y1}, {x: x1+1, y: y1}, {x: x2+sizes[l+1], y: y2}, {x: x2, y: y2}, {x: x1, y: y1},]);
            }
          })
          .style("opacity", d => 0.9*LF/sizes[d.t.length]);

        }
    }

  }


  if (is_wide) {
    fig.append("div")
      .style("position", "absolute")
      .style("top", "0px")
      .style("right", "0px")
      .style("width", "calc((100% - 648px)/2)")
      .style("height", "100%")
      .style("background", "linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,1), rgba(255,255,255,1), rgba(255,255,255,1), rgba(255,255,255,1))");

      fig.append("div")
        .style("position", "absolute")
        .style("top", "0px")
        .style("left", "0px")
        .style("width", "calc((100% - 648px)/4)")
        .style("height", "100%")
        .style("background", "linear-gradient(to left, rgba(255,255,255,0), rgba(255,255,255,1))");
    }

  var stride_ranges = [];
  var size_ranges = [];
  var stride_spans = [];
  var size_spans = [];



    //fig.append('br');
    for (var l = 0; l < max_L; l++){

      if (is_wide){
        var div = fig.append("div");
        if (l==0) div.style("display", "none")
        stride_spans.push(div.append('span')
          .style("position", "absolute")
          .style("top", (30 + 240*(l-1))  + "px")
          .style("right", "200px")
          .style("right", "calc((100% - 648px)/2 - 200px)")
        );
        size_spans.push(div.append('span')
          .style("position", "absolute")
          .style("top", (80 + 240*(l-1))  + "px")
          .style("right", "200px")
          .style("right", "calc((100% - 648px)/2 - 200px)")
        );

        stride_ranges.push(div.append('input').attr('type', 'range')
          .attr('min', 1).attr('max', max_sizes[l])
          .style("position", "absolute")
          .style("top", (20 + 240*(l-1)) + "px")
          .style("right", "200px")
          .style("right", "calc((100% - 648px)/2 - 200px)")
        );
        size_ranges.push(div.append('input')
          .attr('type', 'range').attr('min', 1).attr('max', max_sizes[l])
          .style("position", "absolute")
          .style("top", (70 + 240*(l-1))  + "px")
          .style("right", "200px")
          .style("right", "calc((100% - 648px)/2 - 200px)")
        );
      } else {
        var div = fig.append("div").style({float: "left", "margin": 0, "margin-right" : "20px"});
        if (l==0) div.style("display", "none")
        stride_ranges.push(div.append('input').attr('type', 'range')
          .attr('min', 1).attr('max', max_sizes[l])
        );
        div.append("br");
        stride_spans.push(div.append('span'));
        div.append("br");
        size_ranges.push(div.append('input')
          .attr('type', 'range').attr('min', 1).attr('max', max_sizes[l])
        );
        div.append("br");
        size_spans.push(div.append('span'));
      }
    }
    if (!is_wide) {
      fig.append("br").style("clear", "left")
    }

  /*//fig.append('br');
  for (var l = 0; l < max_L; l++){
    var div = fig.append("div");
    //var div = fig;
    if (l==0) div.style("display", "none")

    stride_spans.push(div.append('span')
      .style("position", "absolute")
      .style("top", (30 + 240*(l-1))  + "px")
      .style("right", "200px")
      .style("right", "calc((100% - 648px)/2 - 200px)")
    );
    size_spans.push(div.append('span')
      .style("position", "absolute")
      .style("top", (80 + 240*(l-1))  + "px")
      .style("right", "200px")
      .style("right", "calc((100% - 648px)/2 - 200px)")
    );

    stride_ranges.push(div.append('input').attr('type', 'range')
      .attr('min', 1).attr('max', max_sizes[l])
      .style("position", "absolute")
      .style("top", (20 + 240*(l-1)) + "px")
      .style("right", "200px")
      .style("right", "calc((100% - 648px)/2 - 200px)")
    );
    size_ranges.push(div.append('input')
      .attr('type', 'range').attr('min', 1).attr('max', max_sizes[l])
      .style("position", "absolute")
      .style("top", (70 + 240*(l-1))  + "px")
      .style("right", "200px")
      .style("right", "calc((100% - 648px)/2 - 200px)")
    );
  }*/

  function update_vis(){
    var strides = [], sizes = [];
    for (var l = 0; l < max_L; l++){
      var stride = parseInt(stride_ranges[l].node().value);
      var size = parseInt(size_ranges[l].node().value);
      strides.push(stride);
      sizes.push(size);
      stride_spans[l].text("stride = " + stride);
      size_spans[l].text("size = " + size);
    }
    draw_vis(sizes, strides);
  }

  var strides = [0,2,2,1];
  var sizes = [5,3,3,3];
  for (var l = 0; l < max_L; l++){
    stride_ranges[l].node().value = strides[l];
    stride_ranges[l].on("input", update_vis);
    size_ranges[l].node().value = sizes[l];
    size_ranges[l].on("input", update_vis);
  }
  update_vis();

}
