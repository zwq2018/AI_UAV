define(function(require, exports, module){
    //用来处理验证码
    $(function(){
        $(document).on("click",".js-verify-refresh",function(){
            var $img=$(".verify-img");
            $img.attr("src",$img.attr("src").replace(/\?\S*/,"?"+Math.random()));
        });
    });
    //渲染验证码的结构
    // target : 目标区域，一个目标区域只有一套验证码结构
    exports.renderVerifyCodeBlock = function( target , url ){
        var html = [
            '<input type="text" maxlength="4" class="verify-code-ipt" placeholder="请输入验证码" />',
            '<a class="verify-img-wrap js-verify-refresh" hidefocus="true" href="javascript:void(0)"><img class="verify-img" src="'+url+'?',Math.random(),'"></a>',
            '<a class="icon-refresh js-verify-refresh" hidefocus="true" href="javascript:void(0)"></a>',
            // '<img class="img-code" src="'+url+'?',Math.random(),'" />',
            // '<span class="verify-code-around">看不清<br/><i>换一换</i></span>',
            '<span class="errtip"></span>'
        ].join("");
        $(target).html(html);
        //验证码
        $( target ).on('fail' ,function(e,msg){
            //console.log(msg);
            /*if($(this).hasClass('fail')){
                return ;
            }else{*/
                $(this).addClass('fail');
                $(this).find('.errtip').text(msg);
            //}
        });
        $( target ).on('succ' ,function(e){
            if($(this).hasClass('fail')){
                $(this).removeClass('fail');
                $(this).find('.errtip').text('');
            }
        });
    };
    //获取用户最终输入的验证码值
    //target ：指定的目标区域
    exports.getResult = function( target ){
        var $elem = $(target).find('.verify-code-ipt');
        if($elem.length == 0){
            return;
        }
        return $.trim($elem.val());
    }

});