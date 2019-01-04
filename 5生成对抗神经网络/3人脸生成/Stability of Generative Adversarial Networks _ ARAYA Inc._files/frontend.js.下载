(function ($) {
  $(function () {
   $('.aps-each-icon').hover(function(){
     var animation_class = $(this).find('.animated').attr('data-animation-class');
     if(animation_class!=='none')
     {
       $(this).find('.animated').addClass(animation_class);
     }
   },function(){
     var animation_class = $(this).find('.animated').attr('data-animation-class');
     if(animation_class!=='none')
     {
       $(this).find('.animated').removeClass(animation_class);
     }
   });
   $('.aps-social-icon-wrapper .aps-each-icon[data-aps-tooltip-enabled="1"]').each(function(i,el){
      var $this=$(el);
      var toolTipText=$this.attr("data-aps-tooltip");
      var toolTipBg=$this.attr("data-aps-tooltip-bg");
      var toolTipTextColor=$this.attr("data-aps-tooltip-color");
      var $toolTipHolder=$this.find('.aps-icon-tooltip');
      $toolTipHolder.text(toolTipText).css({'background-color':toolTipBg,'color':toolTipTextColor,'margin-top':'-'+($toolTipHolder.outerHeight()/2)+'px','margin-left':'-'+($toolTipHolder.outerWidth()/2)+'px'});
      $this.hover(function(){
        $toolTipHolder.stop().fadeIn();
      },function(){
        $toolTipHolder.stop().fadeOut();
      })
   });
   $('.aps-social-icon-wrapper .aps-group-vertical').each(function(){
      var widthArray=new Array();
      $(this).find('img').each(function(i,el){
        var margin=$(el).parents('.aps-each-icon').css('marginLeft').replace('px','');
        var itemWidth = parseInt(($(el).width())+(2*margin));
        widthArray.push(itemWidth);
      });
      widthArray.max=function(){
        return  Math.max.apply(Math,this);
      }
      $(this).width(widthArray.max());
   });
 });
}(jQuery));