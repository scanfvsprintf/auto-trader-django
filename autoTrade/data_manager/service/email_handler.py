# data_manager/service/email_handler.py

import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

logger = logging.getLogger(__name__)

class EmailHandler:
    """
    一个通用的邮件发送处理器。
    它封装了SMTP协议的细节，只向上层提供一个简单的send_email接口。
    所有配置项都在类的起始区域定义，方便统一管理。
    """

    # ==========================================================================
    # SMTP 配置区域 - 请根据您的邮箱服务商填充以下信息
    # 强烈建议在生产环境中使用环境变量或安全的配置管理方式，而非硬编码
    # 对于多数邮箱，您可能需要使用“应用专用密码”而非您的登录密码
    # ==========================================================================
    SMTP_SERVER = 'smtp.qq.com'  # 例如: 'smtp.qq.com' 或 'smtp.gmail.com'
    SMTP_PORT = 465                   # SSL加密端口通常为 465
    SMTP_USER = '876858298@qq.com' # 您的邮箱登录账号
    SMTP_PASSWORD = 'eoyktuuifrmxbdba'  # 您的邮箱授权码或密码
    SENDER_EMAIL = '876858298@qq.com' # 发件人邮箱地址
    SENDER_NAME = '量化交易预案推送'          # 发件人显示名称
    # ==========================================================================

    def send_email(self, recipients: list[str], subject: str, html_content: str) -> bool:
        """
        发送一封HTML格式的邮件给一个或多个收件人。

        :param recipients: 目标邮箱地址的列表, e.g., ['user1@example.com', 'user2@example.com']
        :param subject: 邮件主题
        :param html_content: 邮件正文 (HTML格式)
        :return: True 如果发送成功, False 如果失败
        """
        if not all([self.SMTP_SERVER, self.SMTP_PORT, self.SMTP_USER, self.SMTP_PASSWORD, self.SENDER_EMAIL]):
            logger.critical("SMTP配置不完整，无法发送邮件。请检查 EmailHandler 中的配置项。")
            return False

        if not recipients:
            logger.warning("收件人列表为空，邮件未发送。")
            return False

        # 创建一个带附件的实例
        message = MIMEMultipart('alternative')
        message['From'] = f'"{Header(self.SENDER_NAME, "utf-8").encode()}" <{self.SENDER_EMAIL}>'
        message['To'] = ", ".join(recipients)
        message['Subject'] = Header(subject, 'utf-8')

        # 邮件正文内容
        html_part = MIMEText(html_content, 'html', 'utf-8')
        message.attach(html_part)
        server = None  # 初始化server变量
        try:
            logger.info(f"准备通过 {self.SMTP_SERVER}:{self.SMTP_PORT} 发送邮件至 {recipients}...")
            
            # 1. 手动建立连接
            server = smtplib.SMTP_SSL(self.SMTP_SERVER, self.SMTP_PORT)
            server.login(self.SMTP_USER, self.SMTP_PASSWORD)
            server.sendmail(self.SENDER_EMAIL, recipients, message.as_string())
            
            logger.info(f"邮件发送成功！主题: '{subject}'")
            return True
            
        except smtplib.SMTPException as e:
            logger.error(f"发送邮件时发生SMTP错误: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"发送邮件时发生未知错误: {e}", exc_info=True)
            return False
        finally:
            # 2. 在finally块中确保关闭连接
            if server:
                try:
                    # 3. 对quit()命令进行独立的异常处理
                    server.quit()
                except smtplib.SMTPResponseException as e:
                    # 优雅地处理服务器提前关闭连接的情况
                    logger.warning(f"关闭SMTP连接时发生响应异常 (通常无害): {e}")
                except Exception as e:
                    logger.error(f"关闭SMTP连接时发生未知错误: {e}", exc_info=True)

