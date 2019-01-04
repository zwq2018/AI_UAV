define( function (require, exports, module) {

    /*
        * set : 
        * 参数 1: 要设置cookie的属性;
        * 参数 2: 设置参数1的值;
        * 参数 3: 对象;  { expires : cookie存在天数} 
        
        * get  :
        * 参数 : 传一个要看cookie的name，获取它的值并返回;
     */
    function set (name, value, options)
    {
        // name and value given, set cookie
        if (typeof value != 'undefined')
        {
            options = options || {};
            if (value === null)
            {
                value = '';
                options.expires = -1;
            }
            var expires = '';
            if (options.expires && (typeof options.expires == 'number' || options.expires.toUTCString))
            {
                var date;
                if (typeof options.expires == 'number') {
                    date = new Date();
                    date.setTime(date.getTime() + (options.expires * 24 * 60 * 60 * 1000));
                } else {
                    date = options.expires;
                }
                expires = '; expires=' + date.toUTCString();
            }
            var path   = options.path ? '; path=' + (options.path) : '';
            var domain = options.domain ? '; domain=' + (options.domain) : '';
            var secure = options.secure ? '; secure' : '';
            document.cookie = [name, '=', encodeURIComponent(value), expires, path, domain, secure].join('');
        }
    }

    function get(name)
    {
        var cookieValue = null;
        if (document.cookie && document.cookie !== '')
        {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++)
            {
                var cookie = jQuery.trim(cookies[i]);
                if (cookie.substring(0, name.length + 1) == (name + '='))
                {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
	
	var Cookie = {
		get: get,
		set: set
	}
	
	module.exports = Cookie;
});