#ifndef COMPLEX_MESSAGE_H
#define COMPLEX_MESSAGE_H


#include "message_box.h"

class ComplexMessageBox : public MessageBox {
    Q_OBJECT
    Q_PROPERTY(int opacity MEMBER mOpacity)
  private:
    int mOpacity;
  public:
    explicit ComplexMessageBox(QObject *parent = 0);

};

#endif  // COMPLEX_MESSAGE_H
