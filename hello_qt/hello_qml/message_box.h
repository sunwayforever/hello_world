#ifndef MESSAGE_H
#define MESSAGE_H


#include <QObject>

class MessageBox : public QObject {
    Q_OBJECT

    Q_PROPERTY(int height MEMBER mHeight NOTIFY heightChanged)
    Q_PROPERTY(int width READ width WRITE setWidth)

  public:
    explicit MessageBox(QObject *parent = 0);
    Q_INVOKABLE void foo();

  signals:
    void heightChanged();
    void changedTwoTimes();
    void oopsWidthChanged();

  public slots:
    void onTextChanged(QString);

  private:
    int mHeight;
    int mWidth;
    int width();
    void setWidth(int w);
};

#endif  // MESSAGE_H
