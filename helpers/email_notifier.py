import smtplib

def notify():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("godwinwoo@gmail.com", "")
    msg = "\r\n".join([
        "From: godwinwoo@gmail.com",
        "To: godwinwoo@gmail.com",
        "Subject: python script done",
        "",
        "python script done"
    ])

    server.sendmail("godwinwoo@gmail.com", "godwinwoo@gmail.com", msg)
    server.quit()
