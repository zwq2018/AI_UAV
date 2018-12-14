jQuery(document).ready(function($) {

  // Detect if SVG possible in <img> tags
  var testImg = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNzUiIGhlaWdodD0iMjc1Ij48L3N2Zz4%3D';
  var img = document.createElement('img')
  img.setAttribute('src',testImg);
  
  // This event handler is never called in browsers without SVG support
  img.addEventListener('load',function() {
		// Iterate through all <img> and replace png -> svg  
		$('img.quicklatex-auto-format').attr('src', function() { return $(this).attr('src').replace('.png', '.svg'); })  
  },true);

});