$(document).ready(function () {
    $("#chat-form").on("submit", function (e) {
        e.preventDefault();

        let userInput = $("#user-input").val();
        if (userInput.trim() === "") return;

        // Append user message
        $("#chat-box").append(
            `<div class="message user">${userInput}</div>`
        );

        // Clear input
        $("#user-input").val("");

        // Send AJAX
        $.ajax({
            url: "/get-response/",
            type: "POST",
            data: {
                query: userInput,
                conversation_id: $("#conversation_id").val(),               // <-- added
                csrfmiddlewaretoken: $("input[name=csrfmiddlewaretoken]").val()
            },
            success: function (data) {
                const botHtml = marked.parse(data.bot);
                $("#chat-box").append(`<div class="bot">${botHtml}</div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            },
            error: function (xhr) {
                let msg = "⚠️ Network error, sir.";
                try {
                    const parsed = JSON.parse(xhr.responseText);
                    if (parsed && parsed.error) msg = `⚠️ ${parsed.error}`;
                } catch (_) {}
                $("#chat-box").append(`<div class="message bot">${msg}</div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            }
        });
    });
});
