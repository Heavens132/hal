#include "plugin_management/plugin_arguments_widget.h"

#include "core/interface_cli.h"
#include "core/plugin_manager.h"
#include "core/program_arguments.h"
#include "plugin_management/plugin_schedule_manager.h"

#include <QFormLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QString>

plugin_arguments_widget::plugin_arguments_widget(QWidget* parent) : QFrame(parent), m_form_layout(new QFormLayout())
{
    setLayout(m_form_layout);
}

program_arguments plugin_arguments_widget::get_args()
{
    std::vector<char*> temp_vector;

    QString string = "hal";
    temp_vector.push_back(to_heap_cstring(string));

    for (QPair<QPushButton*, QLineEdit*> pair : m_vector)
    {
        if (pair.first->isChecked())
        {
            string = pair.first->text();
            string.prepend("--");
            temp_vector.push_back(to_heap_cstring(string));

            if (!pair.second->text().isEmpty())
            {
                string = pair.second->text();
                temp_vector.push_back(to_heap_cstring(string));
            }
        }
    }

    int argc    = temp_vector.size();
    char** argv = new char*[argc + 1];

    for (int i = 0; i < argc; i++)
        argv[i] = temp_vector[i];

    argv[argc] = nullptr;

    //    int i;
    //    printf("argc: %d\n", argc);
    //    for(i=0; i < argc; i++) {
    //        printf("argv[%d]: %s\n", i, argv[i]);
    //    }

    program_options options     = m_plugin->get_cli_options();
    program_arguments arguments = options.parse(argc, const_cast<const char**>(argv));

    for (int i = 0; i < argc; ++i)
        delete[] argv[i];

    delete[] argv;

    return arguments;
}

//void plugin_arguments_widget::setup_plugin_layout(const QString& plugin)
//{
//    for (QPair<QPushButton*, QLineEdit*>& pair : m_vector)
//    {
//        pair.first->hide();
//        pair.second->hide();
//        m_form_layout->removeWidget(pair.first);
//        m_form_layout->removeWidget(pair.second);
//        pair.first->deleteLater();
//        pair.second->deleteLater();
//    }
//    m_vector.clear();

//    //USE EXECUTION MANAGER

//    i_factory* factory_ptr = plugin_manager::get_plugin_factory(plugin.toStdString());
//    if (!factory_ptr)
//    {
//        //log_msg(l_error, "failed to get factory for plugin '%s'\n", plugin_name.c_str());
//        return;
//    }

//    m_plugin = std::dynamic_pointer_cast<i_cli>(factory_ptr->query_interface(interface_type::cli));
//    if (!m_plugin)
//    {
//        //log_msg(l_warning, "Plugin %s is not castable to i_cli!\n", plugin_name.c_str());
//        return;
//    }

//    // END

//    for (auto option_tupel : m_plugin->get_cli_options().get_options())
//    {
//        QString name = QString::fromStdString(std::get<0>(option_tupel).at(0));
//        QString description = QString::fromStdString(std::get<1>(option_tupel));
//        QPushButton* button = new QPushButton(name, this);
//        button->setToolTip(description);
//        button->setCheckable(true);
//        QLineEdit* line_edit = new QLineEdit(this);
//        m_vector.append(QPair<QPushButton*, QLineEdit*>(button, line_edit));
//        m_form_layout->addRow(button, line_edit);
//        line_edit->setDisabled(true);
//        connect(button, &QPushButton::toggled, line_edit, &QLineEdit::setEnabled);
//    }
//}

void plugin_arguments_widget::setup(const QString& plugin_name)
{
    for (QPair<QPushButton*, QLineEdit*> pair : m_vector)
    {
        pair.first->deleteLater();
        pair.second->deleteLater();
    }
    m_vector.clear();

    m_plugin = plugin_manager::get_plugin_instance<i_cli>(plugin_name.toStdString(), false);
    if (!m_plugin)
    {
        //log_msg(l_warning, "Plugin %s is not castable to i_cli!\n", plugin_name.c_str());
        return;
    }

    for (auto option_tupel : m_plugin->get_cli_options().get_options())
    {
        QString name        = QString::fromStdString(*std::get<0>(option_tupel).begin());
        QString description = QString::fromStdString(std::get<1>(option_tupel));
        QPushButton* button = new QPushButton(name, this);
        button->setToolTip(description);
        button->setCheckable(true);
        QLineEdit* line_edit = new QLineEdit(this);
        m_vector.append(QPair<QPushButton*, QLineEdit*>(button, line_edit));
        m_form_layout->addRow(button, line_edit);
        line_edit->setDisabled(true);
        connect(button, &QPushButton::toggled, line_edit, &QLineEdit::setEnabled);
        /*line_edit->hide();
        connect(button, &QPushButton::toggled, line_edit, &QLineEdit::setVisible);*/
    }
}

void plugin_arguments_widget::handle_plugin_selected(int index)
{
    for (QPair<QPushButton*, QLineEdit*>& pair : m_vector)
    {
        pair.first->hide();
        pair.second->hide();
        m_form_layout->removeWidget(pair.first);
        m_form_layout->removeWidget(pair.second);
        pair.first->deleteLater();
        pair.second->deleteLater();
    }
    m_vector.clear();

    for (auto arg : plugin_schedule_manager::get_instance()->get_schedule()->at(index).second)
    {
        QPushButton* button = new QPushButton(arg.flag, this);
        button->setToolTip(arg.description);
        button->setCheckable(true);
        button->setChecked(arg.checked);
        QLineEdit* line_edit = new QLineEdit(arg.value, this);
        line_edit->setEnabled(arg.checked);
        m_vector.append(QPair<QPushButton*, QLineEdit*>(button, line_edit));
        m_form_layout->addRow(button, line_edit);
        connect(button, &QPushButton::toggled, line_edit, &QLineEdit::setEnabled);

        connect(button, &QPushButton::clicked, this, &plugin_arguments_widget::handle_button_clicked);
        connect(line_edit, &QLineEdit::textEdited, this, &plugin_arguments_widget::handle_text_edited);
    }
}

void plugin_arguments_widget::handle_button_clicked(bool checked)
{
    QObject* sender     = QObject::sender();
    QPushButton* button = static_cast<QPushButton*>(sender);
    QString flag        = button->text();

    int current_index                     = plugin_schedule_manager::get_instance()->get_current_index();
    QPair<QString, QList<argument>>& pair = (*plugin_schedule_manager::get_instance()->get_schedule())[current_index];
    for (argument& arg : pair.second)
    {
        if (arg.flag == flag)
        {
            arg.checked = checked;
            return;
        }
    }
}

void plugin_arguments_widget::handle_text_edited(const QString& text)
{
    QObject* sender      = QObject::sender();
    QLineEdit* line_edit = static_cast<QLineEdit*>(sender);
    //QString value = line_edit->text();

    QString flag = "";
    for (auto& pair : m_vector)
    {
        if (pair.second == line_edit)
        {
            flag = pair.first->text();
            break;
        }
    }

    QPair<QString, QList<argument>>& pair = (*plugin_schedule_manager::get_instance()->get_schedule())[m_current_index];
    for (argument& arg : pair.second)
    {
        if (arg.flag == flag)
        {
            //arg.value = value;
            arg.value = text;
            return;
        }
    }
}

char* plugin_arguments_widget::to_heap_cstring(const QString& string)
{
    std::string std_string = string.toStdString();
    char* cstring          = new char[std_string.size() + 1];
    std::copy(std_string.begin(), std_string.end(), cstring);
    cstring[std_string.size()] = '\0';
    return cstring;
}
