jQuery(document).ready(function($){

  var siteheight = $("#main_content").height();
  $("#container").css("height", siteheight+"px");

  $("a").bind("focus",function(){if(this.blur)this.blur();});
  $('.rollover').rollover();
  $("a.target_blank").attr("target","_blank");

  $("#comment_area ol > li:even").addClass("even_comment");
  $("#comment_area ol > li:odd").addClass("odd_comment");
  $(".even_comment > .children > li").addClass("even_comment_children");
  $(".odd_comment > .children > li").addClass("odd_comment_children");
  $(".even_comment_children > .children > li").addClass("odd_comment_children");
  $(".odd_comment_children > .children > li").addClass("even_comment_children");
  $(".even_comment_children > .children > li").addClass("odd_comment_children");
  $(".odd_comment_children > .children > li").addClass("even_comment_children");

  $("#trackback_switch").click(function(){
    $("#comment_switch").removeClass("comment_switch_active");
    $(this).addClass("comment_switch_active");
    $("#comment_area").animate({opacity: 'hide'}, 0);
    $("#trackback_area").animate({opacity: 'show'}, 1000);
    return false;
  });

  $("#comment_switch").click(function(){
    $("#trackback_switch").removeClass("comment_switch_active");
    $(this).addClass("comment_switch_active");
    $("#trackback_area").animate({opacity: 'hide'}, 0);
    $("#comment_area").animate({opacity: 'show'}, 1000);
    return false;
  });

 $(".header_menu ul li:has(ul)").addClass("parent_menu");
 $(".header_menu ul li").hover(function(){
  $(">ul:not(:animated)",this).slideDown("fast");
  $(this).addClass("active_menu");
  },
  function(){
  $(">ul",this).slideUp("fast");
  $(this).removeClass("active_menu");
 });

});


jQuery.easing.quart = function (x, t, b, c, d) {
	return -c * ((t=t/d-1)*t*t*t - 1) + b;
};


jQuery(function($){
	var topBtn = $('#return_top');	
	topBtn.hide();
	$(window).scroll(function () {
		if ($(this).scrollTop() > 100) {
			topBtn.fadeIn();
		} else {
			topBtn.fadeOut();
		}
	});
    topBtn.click(function () {
		$('body,html').animate({
			scrollTop: 0
		}, 1000, 'quart');
		return false;
    });
});