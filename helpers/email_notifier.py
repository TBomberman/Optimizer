import smtplib

pw = ""
def notify(message="python script done"):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("godwinwoo2@gmail.com", pw)
    msg = "\r\n".join([
        "From: godwinwoo2@gmail.com",
        "To: godwinwoo@gmail.com",
        "Subject: " + message,
        "",
        message
    ])

    server.sendmail("godwinwoo2@gmail.com", "godwinwoo@gmail.com", msg)
    server.quit()
