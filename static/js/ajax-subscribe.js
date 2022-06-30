(function($) {
    'use strict';

    $('#subscription-form').submit(function(e) {
        e.preventDefault();
        var data = { email: $('#subscribe-input').val() };
        var endpoint = $(this).attr('action');
        $.ajax({
            method: 'POST',
            dataType: "json",
            url: endpoint,
            data: data
        }).done(function(data) {
            if (data.id) {
                alert('Successfully subscribed!');
            } else if (data.title == 'Member Exists') {
                alert('Thanks, but you have alredy subscribed.');
            } else {
                alert('Seems something went wrong. Please Try again.');
            }
        }).fail(function() {
            alert('Seems something went wrong. Please Try again.');
        });
    });

})(jQuery);