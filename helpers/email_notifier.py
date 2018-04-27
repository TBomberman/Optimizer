import smtplib

pw = ""
def notify(message="python script done"):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("godwinwoo@gmail.com", pw)
    msg = "\r\n".join([
        "From: godwinwoo@gmail.com",
        "To: godwinwoo@gmail.com",
        "Subject: " + message,
        "",
        message
    ])

    server.sendmail("godwinwoo@gmail.com", "godwinwoo@gmail.com", msg)
    server.quit()